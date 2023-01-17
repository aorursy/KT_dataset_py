import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

import os

import datetime



plt.rcParams["font.size"] = 14
ts_raw_df = pd.read_csv("/kaggle/input/covid19-dataset-in-japan/covid_jpn_total.csv", parse_dates=["Date"])

ts_raw_df.head()
ts_raw_df.tail()
def sum_nan(x):

    if x.isna().all():

        return np.nan

    else:

        return x.sum()



ts_df = ts_raw_df.groupby("Date").aggregate(sum_nan).asfreq("D")

ts_df.drop(columns="Location", inplace=True)

ts_df.tail()
def plot_diff(col):

    ax = ts_df.loc[:, col].diff().plot(figsize=(16, 3), style=["b."])

    ax.set_title(col)
def plot_cum(col):

    ax = ts_df.loc[:, col].plot(figsize=(16, 3), style=["b."])

    ax.set_title(col)
plot_cum("Positive")
plot_diff("Positive")
plot_cum("Tested")
ts_df["Positive_rate"] = (ts_df["Positive"].diff(7) / ts_df["Tested"].diff(7)).shift(-7)

plot_cum("Positive_rate")

plt.show()
plot_cum("Fatal")
ts_df.loc[:, ["Hosp_require","Hosp_mild","Hosp_severe","Hosp_unknown","Hosp_waiting"]].plot(figsize=(16, 3), style=["."] * 5)

plt.show()
Total = 126166948
ts_df["Hospitalized"] = ts_df[["Hosp_mild","Hosp_severe","Hosp_unknown","Hosp_waiting"]].sum(axis=1, skipna=False)

ts_df[["Hospitalized","Discharged","Fatal"]].plot(figsize=(16, 3), style=["b.", "c.", "g.", "r."])

plt.show()
ts_df["Hosp_require_old"] = ts_df[["Hosp_mild","Hosp_severe","Hosp_unknown","Hosp_waiting"]].sum(axis=1, skipna=False)

ts_df[["Hosp_require_old","Hosp_require"]].plot(figsize=(16, 3), style=["b.","r."])

plt.show()
ts_df["Hosp_require_mod"] = ts_df["Hosp_require_old"].mask(ts_df["Hosp_require_old"].isna(), ts_df["Hosp_require"])

ts_df["Hosp_require_mod"].plot(figsize=(16, 3), style=["b."])

plt.show()
ts_df["Suceptible"] = Total - ts_df[["Positive","Discharged","Fatal"]].sum(axis=1, skipna=False)

ts_df["Suceptible"].plot(figsize=(16, 3), style=["k."])

plt.show()
N_test = 0

i_end_train = len(ts_df) - N_test

ts_stan_df = ts_df[["Suceptible","Positive","Positive_rate","Hosp_require_mod","Discharged","Fatal"]]

ts_stan_df = ts_stan_df.iloc[:i_end_train, :].fillna(-9999, downcast="infer")

ts_stan_df.head()
ts_stan_df.tail()
data = dict(

    N_train=i_end_train,

    N_pred=730,

    Total=Total,

    Detected=ts_stan_df["Positive"].values,

    Hospitalized=ts_stan_df["Hosp_require_mod"].values,

    Discharged=ts_stan_df["Discharged"].values,

    Fatal=ts_stan_df["Fatal"].values,

    Positive_rate=ts_stan_df["Positive_rate"].values,

    N_impute=int((ts_stan_df["Positive_rate"] < 0).sum()),

    i_impute=np.flatnonzero(ts_stan_df["Positive_rate"] < 0) + 1

)

data

date_sim = pd.date_range(ts_df.index.values.min(), periods=data["N_train"] + data["N_pred"])

date_sim
stan_code = '''

functions {

    vector decreasing_simplex(vector x) {

        int N = num_elements(x) + 1;

        vector[N] x_out;

        x_out[1] = 1;

        x_out[2] = x[1];

        for (i in 3:N)

            x_out[i] = x_out[i-1] * x[i-1];

        x_out = x_out / sum(x_out);

        return x_out;

    }



    real poisson_lh(real[] lambda, int start, int end, int[] x) {

        real l = 0;

        int i = 1;

        for (t in start:end) {

            if (x[t] > 0) {

                if (lambda[i] > 0)

                    l += poisson_lpmf(x[t] | lambda[i]);

                else

                    l += -1e10 + lambda[i];

            } else if (x[t] == 0) {

                if (lambda[i] > 0)

                    l += poisson_lpmf(x[t] | lambda[i]);

                else if (lambda[i] < 0)

                    l += -1e10 + lambda[i];

            }

            i += 1;

        }

        return l;

    }



    void smooth_lp(vector x, real v_raw) {

        int N = num_elements(x);

        real v = square(mean(x)) * v_raw;

        x[2:] ~ gamma(square(x[:(N-1)]) / v, x[:(N-1)] / v);

    }

}



data {

    int<lower=1> N_train;

    int<lower=1> N_pred;

    int<lower=1> Total;

    int Detected[N_train];

    int Hospitalized[N_train];

    int Discharged[N_train];

    int Fatal[N_train];

    vector[N_train] Positive_rate;

    int N_impute;

    int<lower=1> i_impute[N_impute];

}



transformed data {

    int count_data[N_train * 4] = append_array(Detected, 

                                    append_array(Hospitalized, 

                                        append_array(Discharged, Fatal)));

    int N_chunk = N_train;



    int max_Delay = 14;

    int max_Delay_I = 35;

    int max_Delay_H = 35;

    

    int N_sim = N_train + 1;

    int N_sim_pred = N_pred + 1;



    real Positive_0 = Detected[1];

    real sum_H0 = Hospitalized[1];

    real L0 = Discharged[1];

    real F0 = 0;



    real detect_by_pcr_min = 1e-3;

    real sum_D0_max = Positive_0 / detect_by_pcr_min;

    real sum_E0_max = sum_D0_max * 1e2;



    real max_Positive_rate = max(Positive_rate);

}



parameters {

    vector<lower=0, upper=1e1>[N_train] contact;



    // ordered mortality with upper limit

    real<lower=1e-2, upper=1> mortality_person_hosp_1;

    real<lower=0, upper=1> rel_mortality_person_I;



    // recovery and transition to more severe state

    simplex[2] recovery_person_exposed;

    simplex[2] recovery_person_detectable;

    simplex[2] recovery_person_infectious;



    simplex[max_Delay] get_detectable_days;



    simplex[max_Delay] get_infectious_days;

    

    simplex[max_Delay_I] mortality_days_infectious;

    simplex[max_Delay_I] hospitalize_days;



    simplex[max_Delay_H] mortality_days_hosp;

    simplex[max_Delay_H] recovery_days_hosp;

    

    real<lower=1, upper=sum_E0_max> sum_E0;

    real<lower=1, upper=sum_D0_max> sum_D0;

    real<lower=1, upper=sum_D0_max> sum_I0;

    

    vector<lower=0, upper=1>[max_Delay - 1] dist_E0_age_dec;

    vector<lower=0, upper=1>[max_Delay - 1] dist_D0_age_dec;

    vector<lower=0, upper=1>[max_Delay_I - 1] dist_I0_age_dec;



    simplex[max_Delay_H] dist_H0_age;



    vector<lower=0, upper=max_Positive_rate>[N_impute] imputation;

    real<lower=0, upper=1> scale_detect;

    real<lower=0> gain_detect;

    real<lower=0, upper=max_Positive_rate> intercept_detect;



    real<lower=1e-1, upper=1> detect_fatality;



    real<lower=1e-8, upper=1e-4> v_contact;

    real<lower=1e-8, upper=1e-4> v_rate_dist;

    real<lower=square(max_Positive_rate)*1e-8, upper=square(max_Positive_rate)*1e-4> v_Positive_rate;

}



transformed parameters {

    

    vector[3] branch_person_infectious;

    vector[2] mortality_person_hosp;

    

    real recovery_exposed = recovery_person_exposed[1] / max_Delay;

    row_vector[max_Delay] get_detectable;



    real recovery_detectable = recovery_person_detectable[1] / max_Delay;

    row_vector[max_Delay] get_infectious;

    

    row_vector[max_Delay_I] mortality_infectious;

    real  recovery_infectious;

    row_vector[max_Delay_I] hospitalize;



    row_vector[max_Delay_H] mortality_hosp;

    row_vector[max_Delay_H] recovery_hosp;



    vector[max_Delay] dist_E0_age = decreasing_simplex(dist_E0_age_dec);

    vector[max_Delay] dist_D0_age = decreasing_simplex(dist_D0_age_dec);

    vector[max_Delay_I] dist_I0_age = decreasing_simplex(dist_I0_age_dec);

    

    row_vector[max_Delay] E0 = (sum_E0 * dist_E0_age)';

    row_vector[max_Delay] D0 = (sum_D0 * dist_D0_age)';

    row_vector[max_Delay_I] I0 = (sum_I0 * dist_I0_age)';

    row_vector[max_Delay_H] H0 = (sum_H0 * dist_H0_age)';



    real R0;



    vector[N_sim] S;

    vector[N_sim] E_out;

    vector[N_sim] D_out;

    vector[N_sim] I_out;

    vector[N_sim] H_out;

    vector[N_sim] L;

    vector[N_sim] R;

    vector[N_sim] F;



    vector[N_sim] Positive;

    vector[N_sim] F_report;



    row_vector[max_Delay] E;

    row_vector[max_Delay] D;

    row_vector[max_Delay_I] I;

    row_vector[max_Delay_H] H;



    vector[N_train] detect;

    vector[N_train] Positive_rate_imputed = Positive_rate;



    for (j in 1:N_impute)

        Positive_rate_imputed[i_impute[j]] = imputation[j];

    detect = scale_detect * inv(1 + exp(gain_detect * (Positive_rate_imputed - intercept_detect)));



    branch_person_infectious[1] = mortality_person_hosp_1 * rel_mortality_person_I;

    branch_person_infectious[2:3] = (1 - branch_person_infectious[1]) * recovery_person_infectious;



    mortality_person_hosp[1] = mortality_person_hosp_1;

    mortality_person_hosp[2] = 1 - mortality_person_hosp_1;

    

    {

        real rel_prob_to_L = recovery_person_exposed[2] * recovery_person_detectable[2] * branch_person_infectious[3] * mortality_person_hosp[2];

        real rel_prob_to_R = recovery_person_exposed[1] + recovery_person_exposed[2] * (recovery_person_detectable[1] + recovery_person_detectable[2] * branch_person_infectious[2]);

        R0 = L0 * rel_prob_to_R / rel_prob_to_L;

    }



    get_detectable = (recovery_person_exposed[2] * get_detectable_days)';



    get_infectious = (recovery_person_detectable[2] * get_infectious_days)';

        

    mortality_infectious = (branch_person_infectious[1] * mortality_days_infectious)';

    recovery_infectious = branch_person_infectious[2] / max_Delay_I;

    hospitalize = (branch_person_infectious[3] * hospitalize_days)';



    mortality_hosp = (mortality_person_hosp[1] * mortality_days_hosp)';

    recovery_hosp = (mortality_person_hosp[2] * recovery_days_hosp)';

 

    //time evolution

    {

        real flux_SE;

        row_vector[max_Delay] flux_EE;

        row_vector[max_Delay] flux_EF;

        row_vector[max_Delay] flux_ER;

        row_vector[max_Delay] flux_ED;



        row_vector[max_Delay] flux_DD;

        row_vector[max_Delay] flux_DF;

        row_vector[max_Delay] flux_DR;

        row_vector[max_Delay] flux_DI;

        

        row_vector[max_Delay_I] flux_II;

        row_vector[max_Delay_I] flux_IF;

        row_vector[max_Delay_I] flux_IR;

        row_vector[max_Delay_I] flux_IH;



        row_vector[max_Delay_H] flux_HH;

        row_vector[max_Delay_H] flux_HF;

        row_vector[max_Delay_H] flux_HL;

        

        real sum_flux_IF;

        real sum_flux_HF;



        real sum_E;

        real sum_D;

        real sum_I;

        

        S[1] = Total - (sum_E0 + sum_D0 + sum_I0 + sum_H0 + L0 + R0 + F0);

        E = E0;

        D = D0;

        I = I0;

        H = H0;

        L[1] = L0;

        R[1] = R0;

        F[1] = F0;

        

        Positive[1] = Positive_0;

        F_report[1] = Fatal[1];



        for (t in 2:N_sim) {

            flux_SE = S[t-1]*(sum(I) + sum(H))*contact[t-1] / Total;



            flux_ED = E .* get_detectable;

            Positive[t] = Positive[t-1] + sum(flux_ED)*detect[t-1];

            flux_ER = E * recovery_exposed;



            flux_DI = D .* get_infectious;

            flux_DR = D * recovery_detectable;

            

            flux_IR = I * recovery_infectious;

            flux_IH = I .* hospitalize;

            flux_IF = I .* mortality_infectious;



            flux_HL = H .* recovery_hosp;

            flux_HF = H .* mortality_hosp;

            

            sum_E = sum(E);

            sum_D = sum(D);

            sum_I = sum(I);

            

            S[t] = S[t-1] - flux_SE;



            flux_EE = E - flux_ED - flux_ER;

            E[1] = flux_SE;

            E[2:max_Delay] = flux_EE[1:(max_Delay-1)];

            E[max_Delay] += flux_EE[max_Delay];

            E_out[t] = sum_E;



            flux_DD = D - flux_DI - flux_DR;

            D[1] = sum(flux_ED);

            D[2:max_Delay] = flux_DD[1:(max_Delay-1)];

            D[max_Delay] += flux_DD[max_Delay];

            D_out[t] = sum_D;



            flux_II = I - flux_IH - flux_IR - flux_IF;

            I[1] = sum(flux_DI);

            I[2:max_Delay_I] = flux_II[1:(max_Delay_I-1)];

            I[max_Delay_I] += flux_II[max_Delay_I];

            I_out[t] = sum_I;

            

            flux_HH = H - flux_HL - flux_HF;

            H[1] = sum(flux_IH);

            H[2:max_Delay_H] = flux_HH[1:(max_Delay_H-1)];

            H[max_Delay_H] += flux_HH[max_Delay_H];

            H_out[t] = sum(H);

            

            R[t] = R[t-1] + sum(flux_ER) + sum(flux_DR) + sum(flux_IR);



            L[t] = L[t-1] + sum(flux_HL);

            

            sum_flux_IF = sum(flux_IF);

            sum_flux_HF = sum(flux_HF);

            F[t] = F[t-1] + sum_flux_IF + sum_flux_HF;

            F_report[t] = F_report[t-1] + sum_flux_IF*detect_fatality + sum_flux_HF;

        }

    }

}





model {

    real estimate[N_train * 5] = to_array_1d(

                                    append_row(Positive[2:N_sim],

                                        append_row(H_out[2:N_sim],

                                            append_row(L[2:N_sim], F_report[2:N_sim]))));



    target += reduce_sum(poisson_lh, estimate, N_chunk, count_data);



    smooth_lp(contact, v_contact);



    smooth_lp(get_detectable_days, v_rate_dist);

    smooth_lp(get_infectious_days, v_rate_dist);

    smooth_lp(hospitalize_days, v_rate_dist);

    smooth_lp(mortality_days_infectious, v_rate_dist);



    for (j in 1:N_impute) {

        Positive_rate_imputed[i_impute[j]] ~ gamma(square(Positive_rate_imputed[i_impute[j]-1]) / v_Positive_rate,

                                                    Positive_rate_imputed[i_impute[j]-1] / v_Positive_rate);

        if ((i_impute[j] < N_train) && (i_impute[j+1] != (i_impute[j] + 1)))

            Positive_rate_imputed[i_impute[j]+1] ~ gamma(square(Positive_rate_imputed[i_impute[j]]) / v_Positive_rate,

                                                        Positive_rate_imputed[i_impute[j]] / v_Positive_rate);

    }

}



generated quantities {

    vector[N_sim_pred] S_pred;

    vector[N_sim_pred] E_out_pred;

    vector[N_sim_pred] D_out_pred;

    vector[N_sim_pred] I_out_pred;

    vector[N_sim_pred] H_out_pred;

    vector[N_sim_pred] L_pred;

    vector[N_sim_pred] R_pred;

    vector[N_sim_pred] F_pred;



    vector[N_sim_pred] Positive_pred;

    vector[N_sim_pred] F_report_pred;



    real mean_duration_IH;

    vector[N_train] reproduction;



    {

        real contact_pred = contact[N_train];

        real detect_pred = detect[N_train];



        row_vector[max_Delay] E_pred = E;

        row_vector[max_Delay] D_pred = D;

        row_vector[max_Delay_I] I_pred = I;

        row_vector[max_Delay_H] H_pred = H;

        

        real flux_SE;

        row_vector[max_Delay] flux_EE;

        row_vector[max_Delay] flux_EF;

        row_vector[max_Delay] flux_ER;

        row_vector[max_Delay] flux_ED;



        row_vector[max_Delay] flux_DD;

        row_vector[max_Delay] flux_DF;

        row_vector[max_Delay] flux_DR;

        row_vector[max_Delay] flux_DI;

        

        row_vector[max_Delay_I] flux_II;

        row_vector[max_Delay_I] flux_IF;

        row_vector[max_Delay_I] flux_IR;

        row_vector[max_Delay_I] flux_IH;



        row_vector[max_Delay_H] flux_HH;

        row_vector[max_Delay_H] flux_HF;

        row_vector[max_Delay_H] flux_HL;

        

        real sum_flux_IF;

        real sum_flux_HF;

        

        real sum_E;

        real sum_D;

        real sum_I;



        S_pred[1] = S[N_sim];

        L_pred[1] = L[N_sim];

        R_pred[1] = R[N_sim];

        F_pred[1] = F[N_sim];

        

        Positive_pred[1] = Positive[N_sim];

        F_report_pred[1] = F_report[N_sim];



        for (t in 2:N_sim_pred) {

            

            flux_SE = S_pred[t-1]*(sum(I_pred) + sum(H_pred))*contact_pred / Total;



            flux_ED = E_pred .* get_detectable;

            Positive_pred[t] = Positive_pred[t-1] + sum(flux_ED)*detect_pred;

            flux_ER = E_pred * recovery_exposed;



            flux_DI = D_pred .* get_infectious;

            flux_DR = D_pred * recovery_detectable;

            

            flux_IR = I_pred * recovery_infectious;

            flux_IH = I_pred .* hospitalize;

            flux_IF = I_pred .* mortality_infectious;



            flux_HL = H_pred .* recovery_hosp;

            flux_HF = H_pred .* mortality_hosp;



            sum_E = sum(E_pred);

            sum_D = sum(D_pred);

            sum_I = sum(I_pred);

            

            S_pred[t] = S_pred[t-1] - flux_SE;



            flux_EE = E_pred - flux_ED - flux_ER;

            E_pred[1] = flux_SE;

            E_pred[2:max_Delay] = flux_EE[1:(max_Delay-1)];

            E_pred[max_Delay] += flux_EE[max_Delay];

            E_out_pred[t] = sum_E;



            flux_DD = D_pred - flux_DI - flux_DR;

            D_pred[1] = sum(flux_ED);

            D_pred[2:max_Delay] = flux_DD[1:(max_Delay-1)];

            D_pred[max_Delay] += flux_DD[max_Delay];

            D_out_pred[t] = sum_D;



            flux_II = I_pred - flux_IH - flux_IR - flux_IF;

            I_pred[1] = sum(flux_DI);

            I_pred[2:max_Delay_I] = flux_II[1:(max_Delay_I-1)];

            I_pred[max_Delay_I] += flux_II[max_Delay_I];

            I_out_pred[t] = sum_I;

            

            flux_HH = H_pred - flux_HL - flux_HF;

            H_pred[1] = sum(flux_IH);

            H_pred[2:max_Delay_H] = flux_HH[1:(max_Delay_H-1)];

            H_pred[max_Delay_H] += flux_HH[max_Delay_H];

            H_out_pred[t] = sum(H_pred);

            

            R_pred[t] = R_pred[t-1] + sum(flux_ER) + sum(flux_DR) + sum(flux_IR);

            L_pred[t] = L_pred[t-1] + sum(flux_HL);

            

            sum_flux_IF = sum(flux_IF);

            sum_flux_HF = sum(flux_HF);

            F_pred[t] = F_pred[t-1] + sum_flux_IF + sum_flux_HF;

            F_report_pred[t] = F_report_pred[t-1] + sum_flux_IF*detect_fatality + sum_flux_HF;

        }

    }



    // average duration in I and H until recovery or die

    // for all the domestically exposed patients

    {

        real numerator = 0;

        real denominator = 0;

        real p_get_infectious = recovery_person_exposed[2] * recovery_person_detectable[2];

        real p_I;

        real p_H;

        for (d_I in 1:max_Delay_I) {

            p_I = recovery_infectious + mortality_infectious[d_I];

            numerator += d_I * p_I;

            denominator += p_I;

            for (d_H in 1:max_Delay_H) {

                p_H = hospitalize[d_I] * (recovery_hosp[d_H] + mortality_hosp[d_H]);

                numerator += (d_I + d_H) * p_H;

                denominator += p_H;

            }

        }

        numerator *= p_get_infectious;

        denominator *= p_get_infectious;

        // probability for not getting infectious

        denominator += recovery_person_exposed[1] + recovery_person_exposed[2] * recovery_person_detectable[1];

        mean_duration_IH = numerator / denominator;

    }

    reproduction = (S[1:N_train] / Total) .* contact * mean_duration_IH;

}



'''
with open("model.stan", mode='w') as f:

    f.write(stan_code)
import sys

!{sys.executable} -m pip install -U cmdstanpy ujson
import cmdstanpy

cmdstanpy.install_cmdstan()
model = cmdstanpy.CmdStanModel(stan_file="model.stan")
start = datetime.datetime.now()

print(start)

os.environ["STAN_NUM_THREADS"] = "4"

try:

    inference = model.variational(data=data, algorithm="fullrank", grad_samples=32, iter=1000000, output_dir="./", save_diagnostics=False)

except Exception as e:

    print(e)

finally:

    print(datetime.datetime.now() - start)
from glob import glob

fn_stan = "model"

stdout_fns = [(f, os.path.getmtime(f)) for f in glob("{}*-stdout.txt".format(fn_stan))]

latest_stdout_fn = sorted(stdout_fns, key=lambda files: files[1])[-1]

print(latest_stdout_fn[0])

elbo_df = pd.read_table(latest_stdout_fn[0], engine="python", skiprows=48, skipfooter=3, sep="\s{2,}", skipinitialspace=True, index_col="iter")

elbo_df.tail()
ax = elbo_df["ELBO"].plot(logy="sym", style=".", ms=2, figsize=(15, 5))

ax.set_ylabel("ELBO")

plt.show()
fns = [(f, os.path.getmtime(f)) for f in glob("{}*.csv".format(fn_stan))]

latest_fn = sorted(fns, key=lambda files: files[1])[-1]

print(latest_fn[0])

inference_df = pd.read_csv(latest_fn[0], engine="python", comment="#")

inference_df.head()
par_names = []

for n in inference_df.columns.tolist():

    if ("." in n):

        par_names.append(n[0:n.find(".")])

    else:

        par_names.append(n)

par_names = set(par_names)        

par_names
par_dim = {}

for name in par_names:

    if name.endswith("_raw") or name.startswith(("lp__")):

        continue

    dim_sample = 0

    for n in inference_df.columns.tolist():

        dim_sample += n.startswith(name + ".")

    if dim_sample == 0:

        dim_sample = 1

    

    par_dim[name] = dim_sample 

        

print(par_dim)
name_hist = []

for p, d in par_dim.items():

    if d <= 3:

        name_hist.append(p)

name_hist
name_hist = [

    'recovery_person_exposed',

    'recovery_person_detectable',

    'branch_person_infectious',

    'mortality_person_hosp',

    'mean_duration_IH',

    'sum_E0',

    'sum_D0',

    'sum_I0',

    'R0',

    'detect_fatality',

    'scale_detect',

    'gain_detect',

    'intercept_detect',

    'v_rate_dist',

    'v_contact',

    'v_Positive_rate',

]
from matplotlib.ticker import ScalarFormatter



n_panel = 0

for name in name_hist:

    n_panel += par_dim[name]

n_rows = int(math.ceil(n_panel / 4))

fig, ax_mat = plt.subplots(nrows=n_rows, ncols=4, figsize=(16, 4*n_rows))

ax = np.ravel(ax_mat)



i = 0

for name in name_hist:

    if par_dim[name] == 1:

        sample = inference_df[name]

        ax[i].hist(sample, bins=50)

        ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax[i].ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))

        ax[i].set_title(name, fontsize=14)

        i += 1

    else:

        for j in range(1, par_dim[name] + 1):

            name_j = name + "." + str(j)

            sample = inference_df[name_j]

            ax[i].hist(sample, bins=50)

            ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            ax[i].ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))

            ax[i].set_title(name_j, fontsize=14)

            i += 1

fig.subplots_adjust(wspace=0.3, hspace=0.4)
name_age = []

for p, d in par_dim.items():

    if (3 < d) & (d < data["N_train"]) & ("dist_" not in p) & ("_days" not in p):

        name_age.append(p)

name_age
name_age = [

'mortality_infectious',

'mortality_hosp',

'recovery_hosp',

'get_detectable',

'get_infectious',

'hospitalize',

'E0',

'D0',

'I0',

'H0',

]
sample_dic = dict()

for name in name_age:

    sample = []

    for j in range(1, par_dim[name] + 1):

        sample.append(inference_df[name + "." + str(j)])

    sample_dic[name] = np.column_stack(sample)
nrows = math.ceil(len(name_age) / 2)

fig, ax_mat = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 3.5*nrows))

ax = np.ravel(ax_mat)

for i, name in enumerate(name_age):

    sample = sample_dic[name]

    sns.boxplot(data=sample, ax=ax[i], color="dodgerblue", linewidth=1, fliersize=1)

    ax[i].set_xticks(np.arange(0, sample.shape[1], 5))

    ax[i].set_xticklabels(np.arange(1, sample.shape[1]+1, 5))

    ax[i].set_title(name)

fig.subplots_adjust(wspace=0.2, hspace=0.4)

fig.suptitle("days from transition to the state", x=0.5, y=0.1)

plt.show()
name_ts = []

# N_sim = data["N_train"] + data["N_pred"] + 1

for p, d in par_dim.items():

    if (d >= data["N_train"]):

        name_ts.append(p)

name_ts
name_ts = [

 'S',

 'E_out',

 'D_out',

 'I_out',

 'H_out',

 'L',

 'R',

 'F',

 'Positive',

 'Positive_rate_imputed',

 'detect',

 'F_report',

 'contact',

 'reproduction',

]
q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

q_ts_dic = dict()

for name in name_ts:

    sample = []

    for j in range(2, par_dim[name] + 1):

        sample.append(inference_df[name + "." + str(j)])

    sample = np.column_stack(sample)



    if name not in ['contact', 'reproduction','Positive_rate_imputed','detect']:

        name_pred = name + "_pred"

        sample_pred = []

        for j in range(2, par_dim[name_pred] + 1):

            sample_pred.append(inference_df[name_pred + "." + str(j)])

        sample_pred = np.column_stack(sample_pred)

        sample = np.hstack([sample, sample_pred])



    q_ts_dic[name] = np.nanquantile(sample, q, axis=0)
q_ts_dic["S"].shape
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["S"][i_list[0][0], :], y2=q_ts_dic["S"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["S"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Suceptible")

# ax.set_ylim(ts_df["Suceptible"].min() * 0.5, ts_df["Suceptible"].max() * 1.025)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))

v = "Positive"

v_data = v

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Cumulative PCR-positive (Domestic)")

ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "Positive"

v_data = v



for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Cumulative PCR-positive (Domestic)")

# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["E_out"][i_list[0][0], :], y2=q_ts_dic["E_out"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["E_out"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Exposed (not detectable or infectious)")

# ax.set_ylim(q_ts_dic["E_out"][:data["N_train"]].min() * 0.1, q_ts_dic["E_out"][:data["N_train"]].max() * 10)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["E_out"][i_list[0][0], :], y2=q_ts_dic["E_out"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["E_out"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Exposed (not detectable or infectious)")

ax.set_ylim(q_ts_dic["E_out"][:data["N_train"]].min() * 0.1, q_ts_dic["E_out"][:data["N_train"]].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["D_out"][i_list[0][0], :], y2=q_ts_dic["D_out"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["D_out"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("PCR-detectable")

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["D_out"][i_list[0][0], :], y2=q_ts_dic["D_out"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["D_out"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("PCR-detectable")

ax.set_ylim(0, q_ts_dic["D_out"][4, :data["N_train"]].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["I_out"][i_list[0][0], :], y2=q_ts_dic["I_out"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["I_out"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Infectious")

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["I_out"][i_list[0][0], :], y2=q_ts_dic["I_out"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["I_out"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Infectious")

ax.set_ylim(0, q_ts_dic["I_out"][4, :data["N_train"]].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["R"][i_list[0][0], :], y2=q_ts_dic["R"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["R"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Cumulative Recovery without Hospitalization")

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic["R"][i_list[0][0], :], y2=q_ts_dic["R"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic["R"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Cumulative Recovery without Hospitalization")

ax.set_ylim(0, q_ts_dic["R"][2, :data["N_train"]].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "H_out"

v_data = "Hosp_require_mod"

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Hospitalized")

ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "H_out"

v_data = "Hosp_require_mod"

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Hospitalized")

# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "L"

v_data = "Discharged"

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Discharged")

ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "L"

v_data = "Discharged"

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Discharged")

# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "F_report"

v_data = "Fatal"

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Fatal")

ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "F_report"

v_data = "Fatal"

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Fatal")

# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "F"

v_data = "Fatal"

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="darkcyan", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Fatal")

ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "F"

v_data = "Fatal"

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="darkcyan", label="Median")

ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported Fatality")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Fatal")

# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["reproduction"][i_list[0][0], :], y2=q_ts_dic["reproduction"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim[1:data["N_train"]], q_ts_dic["reproduction"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.axhline(1, color="blue", lw=1, label="Threshold (1.0)")

ax.set_title("Effective Reproduction Number")

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["reproduction"][i_list[0][0], :], y2=q_ts_dic["reproduction"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim[1:data["N_train"]], q_ts_dic["reproduction"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.axhline(1, color="blue", lw=1, label="Threshold (1.0)")

ax.set_title("Effective Reproduction Number")

ax.set_xlim(date_sim.min() - np.timedelta64(1, 'D'), date_sim[:(data["N_train"]-1)].max() + np.timedelta64(2, 'D'))

ax.tick_params(axis='x', labelrotation=90)

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["contact"][i_list[0][0], :], y2=q_ts_dic["contact"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim[1:data["N_train"]], q_ts_dic["contact"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Effective Contact Rate (per infectious or hospitalized patient per day)")

ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["contact"][i_list[0][0], :], y2=q_ts_dic["contact"][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim[1:data["N_train"]], q_ts_dic["contact"][2, :], "-", color="darkcyan", label="Median")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Effective Contact Rate (per infectious or hospitalized patinet per day)")

ax.set_xlim(date_sim.min() - np.timedelta64(1, 'D'), date_sim[:(data["N_train"]-1)].max() + np.timedelta64(2, 'D'))

ax.tick_params(axis='x', labelrotation=90)

ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

plt.show()
fig, ax = plt.subplots(figsize=(16, 4))



v = "detect"

v_data = "Positive_rate"

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():

    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)

ax.plot(date_sim[1:data["N_train"]], q_ts_dic[v][2, :], "-", color="darkcyan", label="Median")

ax_r = ax.twinx()

ax_r.plot(ts_df.index.values, ts_df[v_data], "k.", label="Positive rate (right axis)")



ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")

ax.set_title("Detection Rate with PCR")

ax.set_xlim(date_sim.min() - np.timedelta64(1, 'D'), date_sim[:(data["N_train"]-1)].max() + np.timedelta64(2, 'D'))

ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

ax_r.legend(loc="upper left", bbox_to_anchor=(1.05, 0.5))

plt.show()