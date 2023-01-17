from datetime import datetime

time_format = "%d%b%Y %H:%M"

datetime.now().strftime(time_format)
from datetime import timedelta

from dateutil.relativedelta import relativedelta

import os

from pprint import pprint

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib.ticker import ScalarFormatter

%matplotlib inline

import numpy as np

import optuna

optuna.logging.disable_default_handler()

import pandas as pd

pd.plotting.register_matplotlib_converters()

import seaborn as sns

from scipy.integrate import solve_ivp
np.random.seed(2019)

os.environ["PYTHONHASHSEED"] = "2019"
plt.style.use("seaborn-ticks")

plt.rcParams["xtick.direction"] = "in"

plt.rcParams["ytick.direction"] = "in"

plt.rcParams["font.size"] = 11.5

plt.rcParams["figure.figsize"] = (9, 6)
population_date = "06Mar2020"

_dict = {

    "Global": "7 738 323 220",

    "China": "1 405 371 596",

    "Japan": "125 406 227",

    "South Korea": "51 277 160",

    "Italy": "59 813 196",

    "Iran": "83 473 631",

}

population_dict = {k: int(v.replace(" ", "")) for (k, v) in _dict.items()}

df = pd.io.json.json_normalize(population_dict)

df.index = [f"Total population on {population_date}"]

df
for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def line_plot(df, title, ylabel="Cases", h=None, v=None, xlim=(None, None), ylim=(0, None), math_scale=True):

    """

    Show chlonological change of the data.

    """

    ax = df.plot()

    if math_scale:

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))

    ax.set_title(title)

    ax.set_xlabel(None)

    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)

    ax.set_ylim(*ylim)

    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)

    if h is not None:

        ax.axhline(y=h, color="black", linestyle="--")

    if v is not None:

        ax.axvline(x=v, color="black", linestyle="--")

    plt.tight_layout()

    plt.show()
def create_target_df(ncov_df, total_population, places=None, excluded_places=None):

    """

    Select the records of the palces, calculate the number of susceptible people,

     and calculate the elapsed time [day] from the start date of the target dataframe.

    @ncov_df <pd.DataFrame>: the clean data

    @total_population <int>: total population in the places

    @places <list[tuple(<str/None>, <str/None>)]: the list of places

        - if the list is None, all data will be used

        - (str, str): both of country and province are specified

        - (str, None): only country is specified

        - (None, str) or (None, None): Error

    @excluded_places <list[tuple(<str/None>, <str/None>)]: the list of excluded places

        - if the list is None, all data in the "places" will be used

        - (str, str): both of country and province are specified

        - (str, None): only country is specified

        - (None, str) or (None, None): Error

    @return <tuple(2 objects)>:

        - 1. start_date <pd.Timestamp>: the start date of the selected records

        - 2. target_df <pd.DataFrame>:

            - column T: elapsed time [min] from the start date of the dataset

            - column Susceptible: the number of patients who are in the palces but not infected/recovered/died

            - column Infected: the number of infected cases

            - column Recovered: the number of recovered cases

            - column Deaths: the number of death cases

    """

    # Select the target records

    df = ncov_df.copy()

    c_series = ncov_df["Country"]

    p_series = ncov_df["Province"]

    if places is not None:

        df = pd.DataFrame(columns=ncov_df.columns)

        for (c, p) in places:

            if c is None:

                raise Exception("places: Country must be specified!")

            if p is None:

                new_df = ncov_df.loc[c_series == c, :]

            else:

                new_df = ncov_df.loc[(c_series == c) & (p_series == p), :]

            df = pd.concat([df, new_df], axis=0)

    if excluded_places is not None:

        for (c, p) in excluded_places:

            if c is None:

                raise Exception("excluded_places: Country must be specified!")

            if p is None:

                df = df.loc[c_series != c, :]

            else:

                c_df = df.loc[(c_series == c) & (p_series != p), :]

                other_df = df.loc[c_series != c, :]

                df = pd.concat([c_df, other_df], axis=0)

    df = df.groupby("Date").sum().reset_index()

    start_date = df.loc[df.index[0], "Date"]

    # column T

    df["T"] = ((df["Date"] - start_date).dt.total_seconds() / 60).astype(int)

    # coluns except T

    df["Susceptible"] = total_population - df["Infected"] - df["Recovered"] - df["Deaths"]

    response_variables = ["Susceptible", "Infected", "Recovered", "Deaths"]

    # Return

    target_df = df.loc[:, ["T", *response_variables]]

    return (start_date, target_df)
def simulation(model, initials, step_n, **params):

    """

    Solve ODE of the model.

    @model <ModelBase>: the model

    @initials <tuple[float]>: the initial values

    @step_n <int>: the number of steps

    """

    tstart, dt, tend = 0, 1, step_n

    sol = solve_ivp(

        fun=model(**params),

        # Implicit Runge-Kutta method of the Radau IIA family of order 5

        # method="Radau",

        t_span=[tstart, tend],

        y0=np.array(initials, dtype=np.float64),

        t_eval=np.arange(tstart, tend + dt, dt)

    )

    t_df = pd.Series(data=sol["t"], name="t")

    y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=model.VARIABLES)

    sim_df = pd.concat([t_df, y_df], axis=1)

    return sim_df
class Estimater(object):

    def __init__(self, model, ncov_df, total_population, places=None, excluded_places=None):

        """

        Set training data.

        @model <ModelBase>: the model

        @the other params: See the function named create_target_df()

        """

        dataset = model.create_dataset(

            ncov_df, total_population, places=places, excluded_places=excluded_places

        )

        self.start_time, self.initials, self.Tend, self.train_df = dataset

        self.total_population = total_population

        self.model = model

        self.param_dict = model.param_dict()

        self.study = None

        self.optimize_df = None



    def run(self, n_trials=700):

        """

        Try estimation (optimization of parameters and tau).

        @n_trials <int>: the number of trials

        """

        if self.study is None:

            self.study = optuna.create_study(direction="minimize")

        self.study.optimize(

            lambda x: self.objective(x),

            n_trials=n_trials,

            n_jobs=-1

        )

        param_dict = self.study.best_params.copy()

        param_dict["R0"] = self.calc_r0()

        param_dict["score"] = self.score()

        param_dict.update(self.calc_days_dict())

        self.param_dict = param_dict.copy()

        return param_dict



    def history_df(self):

        """

        Return the hsitory of optimization.

        @return <pd.DataFrame>

        """

        optimize_df = self.study.trials_dataframe()

        optimize_df["time[s]"] = optimize_df["datetime_complete"] - optimize_df["datetime_start"]

        optimize_df["time[s]"] = optimize_df["time[s]"].dt.total_seconds()

        self.optimize_df = optimize_df.drop(["datetime_complete", "datetime_start"], axis=1)

        return self.optimize_df.sort_values("value", ascending=True)



    def history_graph(self):

        """

        Show the history of parameter search using pair-plot.

        """

        if self.optimize_df is None:

            self.history_df()

        df = self.optimize_df.copy()

        sns.pairplot(df.loc[:, df.columns.str.startswith("params_")], diag_kind="kde", markers="+")

        plt.show()



    def objective(self, trial):

        # Time

        tau = trial.suggest_int("tau", 1, 1440)

        train_df_divided = self.train_df.copy()

        train_df_divided["t"] = (train_df_divided["T"] / tau).astype(int)

        # Parameters

        p_dict = dict()

        for (name, info) in self.param_dict.items():

            if info[0] == "float":

                param = trial.suggest_uniform(name, info[1], info[2])

            else:

                param = trial.suggest_int(name, info[1], info[2])

            p_dict[name] = param

        # Simulation

        t_end = train_df_divided.loc[train_df_divided.index[-1], "t"]

        sim_df = simulation(self.model, self.initials, step_n=t_end, **p_dict)

        return self.error_f(train_df_divided, sim_df)



    def error_f(self, train_df_divided, sim_df):

        """

        We need to minimize the difference of the observed values and estimated values.

        This function calculate the difference of the estimated value and obsereved value.

        """

        df = pd.merge(train_df_divided, sim_df, on="t", suffixes=("_observed", "_estimated"))

        diffs = [

            # Weighted Average: the recent data is more important

            p * np.average(abs(df[f"{v}_observed"] - df[f"{v}_estimated"]), weights=df["t"])

            for (p, v) in zip(self.model.PRIORITIES, self.model.VARIABLES)

        ]

        return sum(diffs) * self.total_population



    def compare_df(self):

        """

        Show the taining data and simulated data in one dataframe.

        

        """

        est_dict = self.study.best_params.copy()

        tau = est_dict["tau"]

        est_dict.pop("tau")

        observed_df = self.train_df.drop("T", axis=1)

        observed_df["t"] = (self.train_df["T"] / tau).astype(int)

        t_end = observed_df.loc[observed_df.index[-1], "t"]

        sim_df = simulation(self.model, self.initials, step_n=t_end, **est_dict)

        df = pd.merge(observed_df, sim_df, on="t", suffixes=("_observed", "_estimated"))

        df = df.set_index("t")

        return df



    def compare_graph(self):

        """

        Compare obsereved and estimated values in graphs.

        """

        df = self.compare_df()

        val_len = len(self.model.VARIABLES)

        fig, axes = plt.subplots(ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2))

        for (ax, v) in zip(axes.ravel()[1:], self.model.VARIABLES[1:]):

            df[[f"{v}_observed", f"{v}_estimated"]].plot.line(

                ax=ax, ylim=(0, None), sharex=True,

                title=f"{self.model.NAME}: Comparison of observed/estimated {v}(t)"

            )

            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))

            ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)

        for v in self.model.VARIABLES[1:]:

            df[f"{v}_diff"] = df[f"{v}_observed"] - df[f"{v}_estimated"]

            df[f"{v}_diff"].plot.line(

                ax=axes.ravel()[0], sharex=True,

                title=f"{self.model.NAME}: observed - estimated"

            )

        axes.ravel()[0].axhline(y=0, color="black", linestyle="--")

        axes.ravel()[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        axes.ravel()[0].ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))

        axes.ravel()[0].legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)

        fig.tight_layout()

        fig.show()

    

    def calc_r0(self):

        """

        Calculate R0.

        """

        est_dict = self.study.best_params.copy()

        est_dict.pop("tau")

        model_instance = self.model(**est_dict)

        return model_instance.calc_r0()



    def calc_days_dict(self):

        """

        Calculate 1/beta etc.

        """

        est_dict = self.study.best_params.copy()

        tau = est_dict["tau"]

        est_dict.pop("tau")

        model_instance = self.model(**est_dict)

        return model_instance.calc_days_dict(tau)



    def predict_df(self, step_n):

        """

        Predict the values in the future.

        @step_n <int>: the number of steps

        @return <pd.DataFrame>: predicted data for measurable variables.

        """

        est_dict = self.study.best_params.copy()

        tau = est_dict["tau"]

        est_dict.pop("tau")

        df = simulation(self.model, self.initials, step_n=step_n, **est_dict)

        df["Time"] = (df["t"] * tau).apply(lambda x: timedelta(minutes=x)) + self.start_time

        df = df.set_index("Time").drop("t", axis=1)

        df = (df * self.total_population).astype(int)

        upper_cols = [n.upper() for n in df.columns]

        df.columns = upper_cols

        df = self.model.calc_variables_reverse(df).drop(upper_cols, axis=1)

        return df



    def predict_graph(self, step_n, name, excluded_cols=None):

        """

        Predict the values in the future and create a figure.

        @step_n <int>: the number of steps

        @name <str>: place name

        @excluded_cols <list[str]>: the excluded columns in the figure

        """

        df = self.predict_df(step_n=step_n)

        if excluded_cols is not None:

            df = df.drop(excluded_cols, axis=1)

        r0 = self.param_dict["R0"]

        title = f"Prediction in {name} with {self.model.NAME} model: R0 = {r0}"

        line_plot(df, title, v= datetime.today(), h=self.total_population)



    def score(self):

        """

        Return the sum of differences of observed and estimated values devided by the number of steps.

        """

        variables = self.model.VARIABLES[:]

        compare_df = self.compare_df()

        score = 0

        for v in variables:

            score += abs(compare_df[f"{v}_observed"] - compare_df[f"{v}_estimated"]).sum()

        score = score / len(compare_df)

        return score
class ModelBase(object):

    NAME = "Model"

    VARIABLES = ["x"]

    PRIORITIES = np.array([1])



    @classmethod

    def param_dict(cls):

        """

        Define parameters without tau. This function should be overwritten.

        @return <dict[name]=(type, min, max):

            @type <str>: "float" or "int"

            @min <float/int>: min value

            @max <float/int>: max value

        """

        param_dict = dict()

        return param_dict



    @staticmethod

    def calc_variables(df):

        """

        Calculate the variables of the model.

        This function should be overwritten.

        @df <pd.DataFrame>

        @return <pd.DataFrame>

        """

        return df



    @staticmethod

    def calc_variables_reverse(df):

        """

        Calculate measurable variables using the variables of the model.

        This function should be overwritten.

        @df <pd.DataFrame>

        @return <pd.DataFrame>

        """

        return df



    @classmethod

    def create_dataset(cls, ncov_df, total_population, places=None, excluded_places=None):

        """

        Create dataset with the model-specific varibles.

        The variables will be divided by total population.

        The column names (not include T) will be lower letters.

        @params: See the function named create_target_df()

        @return <tuple(objects)>:

            - start_date <pd.Timestamp>

            - initials <tuple(float)>: the initial values

            - Tend <int>: the last value of T

            - df <pd.DataFrame>: the dataset

        """

        start_date, target_df = create_target_df(

            ncov_df, total_population, places=places, excluded_places=excluded_places

        )

        df = cls.calc_variables(target_df).set_index("T") / total_population

        df.columns = [n.lower() for n in df.columns]

        initials = df.iloc[0, :].values

        df = df.reset_index()

        Tend = df.iloc[-1, 0]

        return (start_date, initials, Tend, df)



    def calc_r0(self):

        """

        Calculate R0. This function should be overwritten.

        """

        return None



    def calc_days_dict(self, tau):

        """

        Calculate 1/beta [day] etc.

        This function should be overwritten.

        @param tau <int>: tau value [hour]

        """

        return dict()
class SIR(ModelBase):

    NAME = "SIR"

    VARIABLES = ["x", "y", "z"]

    PRIORITIES = np.array([1, 1, 1])



    def __init__(self, rho, sigma):

        super().__init__()

        self.rho = float(rho)

        self.sigma = float(sigma)



    def __call__(self, t, X):

        # x, y, z = [X[i] for i in range(len(self.VARIABLES))]

        # dxdt = - self.rho * x * y

        # dydt = self.rho * x * y - self.sigma * y

        # dzdt = self.sigma * y

        dxdt = - self.rho * X[0] * X[1]

        dydt = self.rho * X[0] * X[1] - self.sigma * X[1]

        dzdt = self.sigma * X[1]

        return np.array([dxdt, dydt, dzdt])



    @classmethod

    def param_dict(cls):

        param_dict = super().param_dict()

        param_dict["rho"] = ("float", 0, 1)

        param_dict["sigma"] = ("float", 0, 1)

        return param_dict



    @staticmethod

    def calc_variables(df):

        df["X"] = df["Susceptible"]

        df["Y"] = df["Infected"]

        df["Z"] = df["Recovered"] + df["Deaths"]

        return df.loc[:, ["T", "X", "Y", "Z"]]



    @staticmethod

    def calc_variables_reverse(df):

        df["Susceptible"] = df["X"]

        df["Infected"] = df["Y"]

        df["Recovered/Deaths"] = df["Z"]

        return df



    def calc_r0(self):

        r0 = self.rho / self.sigma

        return round(r0, 2)



    def calc_days_dict(self, tau):

        _dict = dict()

        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)

        _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)

        return _dict
class SIRD(ModelBase):

    NAME = "SIR-D"

    VARIABLES = ["x", "y", "z", "w"]

    PRIORITIES = np.array([1, 10, 10, 1])



    def __init__(self, kappa, rho, sigma):

        super().__init__()

        self.kappa = float(kappa)

        self.rho = float(rho)

        self.sigma = float(sigma)



    def __call__(self, t, X):

        # x, y, z, w = [X[i] for i in range(len(self.VARIABLES))]

        # dxdt = - self.rho * x * y

        # dydt = self.rho * x * y - (self.sigma + self.kappa) * y

        # dzdt = self.sigma * y

        # dwdt = self.kappa * y

        dxdt = - self.rho * X[0] * X[1]

        dydt = self.rho * X[0] * X[1] - (self.sigma + self.kappa) * X[1]

        dzdt = self.sigma * X[1]

        dwdt = self.kappa * X[1]

        return np.array([dxdt, dydt, dzdt, dwdt])



    @classmethod

    def param_dict(cls):

        param_dict = super().param_dict()

        param_dict["kappa"] = ("float", 0, 1)

        param_dict["rho"] = ("float", 0, 1)

        param_dict["sigma"] = ("float", 0, 1)

        return param_dict



    @staticmethod

    def calc_variables(df):

        df["X"] = df["Susceptible"]

        df["Y"] = df["Infected"]

        df["Z"] = df["Recovered"]

        df["W"] = df["Deaths"]

        return df.loc[:, ["T", "X", "Y", "Z", "W"]]



    @staticmethod

    def calc_variables_reverse(df):

        df["Susceptible"] = df["X"]

        df["Infected"] = df["Y"]

        df["Recovered"] = df["Z"]

        df["Deaths"] = df["W"]

        return df



    def calc_r0(self):

        r0 = self.rho / (self.sigma + self.kappa)

        return round(r0, 2)



    def calc_days_dict(self, tau):

        _dict = dict()

        _dict["1/alpha2 [day]"] = int(tau / 24 / 60 / self.kappa)

        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)

        _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)

        return _dict
class SIRF(ModelBase):

    NAME = "SIR-F"

    VARIABLES = ["x", "y", "z", "w"]

    PRIORITIES = np.array([1, 10, 10, 1])



    def __init__(self, theta, kappa, rho, sigma):

        super().__init__()

        self.theta = float(theta)

        self.kappa = float(kappa)

        self.rho = float(rho)

        self.sigma = float(sigma)



    def __call__(self, t, X):

        # x, y, z, w = [X[i] for i in range(len(self.VARIABLES))]

        # dxdt = - self.rho * x * y

        # dydt = self.rho * (1 - self.theta) * x * y - (self.sigma + self.kappa) * y

        # dzdt = self.sigma * y

        # dwdt = self.rho * self.theta * x * y + self.kappa * y

        dxdt = - self.rho * X[0] * X[1]

        dydt = self.rho * (1 - self.theta) * X[0] * X[1] - (self.sigma + self.kappa) * X[1]

        dzdt = self.sigma * X[1]

        dwdt = self.rho * self.theta * X[0] * X[1] + self.kappa * X[1]

        return np.array([dxdt, dydt, dzdt, dwdt])



    @classmethod

    def param_dict(cls):

        param_dict = super().param_dict()

        param_dict["theta"] = ("float", 0, 1)

        param_dict["kappa"] = ("float", 0, 1)

        param_dict["rho"] = ("float", 0, 1)

        param_dict["sigma"] = ("float", 0, 1)

        return param_dict



    @staticmethod

    def calc_variables(df):

        df["X"] = df["Susceptible"]

        df["Y"] = df["Infected"]

        df["Z"] = df["Recovered"]

        df["W"] = df["Deaths"]

        return df.loc[:, ["T", "X", "Y", "Z", "W"]]



    @staticmethod

    def calc_variables_reverse(df):

        df["Susceptible"] = df["X"]

        df["Infected"] = df["Y"]

        df["Recovered"] = df["Z"]

        df["Deaths"] = df["W"]

        return df



    def calc_r0(self):

        r0 = self.rho * (1 - self.theta) / (self.sigma + self.kappa)

        return round(r0, 2)



    def calc_days_dict(self, tau):

        _dict = dict()

        _dict["alpha1 [-]"] = round(self.theta, 2)

        _dict["1/alpha2 [day]"] = int(tau / 24 / 60 / self.kappa)

        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)

        _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)

        return _dict
class SIRFV(ModelBase):

    NAME = "SIR-FV"

    VARIABLES = ["x", "y", "z", "w"]

    PRIORITIES = np.array([1, 10, 10, 1])



    def __init__(self, theta, kappa, rho, sigma, omega=None, n=None, v_per_day=None):

        """

        (n and v_per_day) or omega must be applied.

        @n <float or int>: total population

        @v_par_day <float or int>: vacctinated persons per day

        """

        super().__init__()

        self.theta = float(theta)

        self.kappa = float(kappa)

        self.rho = float(rho)

        self.sigma = float(sigma)

        if omega is None:

            try:

                self.omega = float(v_per_day) / float(n)

            except TypeError:

                s = "Neither (n and va_per_day) nor omega must be applied!"

                raise TypeError(s)

        else:

            self.omega = float(omega)



    def __call__(self, t, X):

        # x, y, z, w = [X[i] for i in range(len(self.VARIABLES))]

        # x with vacctination

        dxdt = - self.rho * X[0] * X[1] - self.omega

        dxdt = 0 - X[0] if X[0] + dxdt < 0 else dxdt

        # y, z, w

        dydt = self.rho * (1 - self.theta) * X[0] * X[1] - (self.sigma + self.kappa) * X[1]

        dzdt = self.sigma * X[1]

        dwdt = self.rho * self.theta * X[0] * X[1] + self.kappa * X[1]

        return np.array([dxdt, dydt, dzdt, dwdt])



    @classmethod

    def param_dict(cls):

        param_dict = super().param_dict()

        param_dict["theta"] = ("float", 0, 1)

        param_dict["kappa"] = ("float", 0, 1)

        param_dict["rho"] = ("float", 0, 1)

        param_dict["sigma"] = ("float", 0, 1)

        param_dict["omega"] = ("float", 0, 1)

        return param_dict



    @staticmethod

    def calc_variables(df):

        df["X"] = df["Susceptible"]

        df["Y"] = df["Infected"]

        df["Z"] = df["Recovered"]

        df["W"] = df["Deaths"]

        return df.loc[:, ["T", "X", "Y", "Z", "W"]]



    @staticmethod

    def calc_variables_reverse(df):

        df["Susceptible"] = df["X"]

        df["Infected"] = df["Y"]

        df["Recovered"] = df["Z"]

        df["Deaths"] = df["W"]

        return df



    def calc_r0(self):

        r0 = self.rho * (1 - self.theta) / (self.sigma + self.kappa)

        return round(r0, 2)



    def calc_days_dict(self, tau):

        _dict = dict()

        _dict["alpha1 [-]"] = round(self.theta, 2)

        _dict["1/alpha2 [day]"] = int(tau / 24 / 60 / self.kappa)

        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)

        _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)

        return _dict
class Predicter(object):

    """

    Predict the future using models.

    """

    def __init__(self, name, estimater, date_format="%d%b%Y"):

        """

        @name <str>: place name

        @estimater <Estimater>: estimater between the start date and today

        @date_format <str>: date format to display in figures

        """

        self.name = name

        self.total_population = estimater.total_population

        self.start_time = estimater.start_time

        self.reverse_f = estimater.model.calc_variables_reverse

        self.date_format = date_format

        first_params = estimater.study.best_params.copy()

        self.tau = first_params["tau"]

        first_params.pop("tau")

        # Set first model (between the start date and today)

        now = datetime.now()

        step_n = int((now - self.start_time).total_seconds() / self.tau / 60)

        r0 = estimater.calc_r0()

        self.sim_df = simulation(

            estimater.model, estimater.initials, step_n=step_n,

            **first_params

        )

        self.info_list = [f"{estimater.model.NAME}({r0}, -{now.strftime(date_format)})"]



    def add(self, model, days, **param_dict):

        """

        @model <ModelBase>: the epidemic model

        @days <int>: the number of days

        @**param_dict <dict>: keyword arguments of the model

        """

        initials = self.sim_df.set_index("t").iloc[-1, :]

        param_dict.pop("tau")

        new_df = simulation(model, initials, step_n=int(days * 24 * 60 / self.tau), **param_dict)

        if self.sim_df.columns.tolist() != new_df.columns.tolist():

            raise Exception(f"The variables must be {', '.join(self.sim_df.columns)}!")

        new_df["t"] = new_df["t"] + len(self.sim_df)

        self.sim_df = pd.concat([self.sim_df, new_df], axis=0)

        r0 = model(**param_dict).calc_r0()

        last_time = self.start_time + timedelta(minutes=(len(self.sim_df) - 1) * self.tau)

        self.info_list.append(f"{model.NAME}({r0}, -{last_time.strftime(self.date_format)})")



    def restore_df(self):

        """

        Return the dimentional simulated data.

        @return <pd.DataFrame>

        """

        df = self.sim_df.copy()

        df["Time"] = self.start_time + df["t"].apply(lambda x: timedelta(minutes=x * self.tau))

        df = df.drop("t", axis=1).set_index("Time") * self.total_population

        df = df.astype(int)

        upper_cols = [n.upper() for n in df.columns]

        df.columns = upper_cols

        df = self.reverse_f(df).drop(upper_cols, axis=1)

        return df



    def restore_graph(self, drop_cols=None):

        """

        Show the dimentional simulate data as a figure.

        @drop_cols <list[str]>: the columns not to be shown

        """

        df = self.restore_df()

        if drop_cols is not None:

            df = df.drop(drop_cols, axis=1)

        info = ", ".join(self.info_list)

        line_plot(

            df,

            title=f"{self.name}: {info}",

            v=datetime.today(), h=self.total_population

        )
raw = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

raw.tail()
raw.info()
raw.describe()
pd.DataFrame(raw.isnull().sum()).T
", ".join(raw["Country/Region"].unique().tolist())
pprint(raw.loc[raw["Country/Region"] == "Others", "Province/State"].unique().tolist(), compact=True)
data_cols = ["Infected", "Deaths", "Recovered"]

rate_cols = ["Fatal per Confirmed", "Recovered per Confirmed", "Fatal per (Fatal or Recovered)"]

variable_dict = {"Susceptible": "S", "Infected": "I", "Recovered": "R", "Deaths": "D"}
ncov_df = raw.rename({"ObservationDate": "Date", "Province/State": "Province"}, axis=1)

ncov_df["Date"] = pd.to_datetime(ncov_df["Date"])

ncov_df["Country"] = ncov_df["Country/Region"].replace({"Mainland China": "China"})

ncov_df["Province"] = ncov_df["Province"].fillna("-").replace({"Cruise Ship": "Diamond Princess cruise ship"})

ncov_df["Infected"] = ncov_df["Confirmed"] - ncov_df["Deaths"] - ncov_df["Recovered"]

ncov_df[data_cols] = ncov_df[data_cols].astype(int)

ncov_df = ncov_df.loc[:, ["Date", "Country", "Province", *data_cols]]

ncov_df.tail()
ncov_df.info()
ncov_df.describe(include="all").fillna("-")
pd.DataFrame(ncov_df.isnull().sum()).T
", ".join(ncov_df["Country"].unique().tolist())
total_df = ncov_df.loc[ncov_df["Country"] != "China", :].groupby("Date").sum()

total_df[rate_cols[0]] = total_df["Deaths"] / total_df[data_cols].sum(axis=1)

total_df[rate_cols[1]] = total_df["Recovered"] / total_df[data_cols].sum(axis=1)

total_df[rate_cols[2]] = total_df["Deaths"] / (total_df["Deaths"] + total_df["Recovered"])

total_df.tail()
f"{(total_df.index.max() - total_df.index.min()).days} days have passed from the start date."
line_plot(total_df[data_cols], "Cases over time (Total except China)")
line_plot(total_df[rate_cols], "Rate over time (Total except China)", ylabel="", math_scale=False)
total_df[rate_cols].plot.kde()

plt.title("Kernel density estimation of the rates (Total except China)")

plt.show()
total_df[rate_cols].describe().T
train_start_date, train_df = create_target_df(

    ncov_df, population_dict["Global"] - population_dict["China"], excluded_places=[("China", None)]

)

train_start_date.strftime(time_format)
train_df.tail()
df = train_df.rename(variable_dict, axis=1)

for (_, v) in variable_dict.items():

    df[f"d{v}/dT"] = df[v].diff() / df["T"].diff()

df.set_index("T").corr().loc[variable_dict.values(), :].style.background_gradient(axis=None)
sns.lmplot(

    x="I", y="value", col="diff", sharex=False, sharey=False,

    data=df[["I", "dI/dT", "dR/dT", "dD/dT"]].melt(id_vars="I", var_name="diff")

)

plt.show()
trend_df = ncov_df.loc[ncov_df["Country"] != "China",["Date", *data_cols]].groupby("Date").sum().reset_index()

trend_df["Confirmed"] = trend_df["Infected"] + trend_df["Deaths"] + trend_df["Recovered"]

trend_df = trend_df.rename({"Date": "ds"}, axis=1)

trend_df = trend_df.loc[:, ["ds", "Confirmed", "Deaths", "Recovered"]]

trend_df = trend_df.set_index("ds").apply(np.log10).reset_index().replace([np.inf, -np.inf], 0)

trend_df.columns = ["ds", "Log10(Confirmed)", "Log10(Deaths)", "Log10(Recovered)"]

trend_df.tail()
df = trend_df.rename({"Log10(Confirmed)": "y"}, axis=1).loc[:, ["ds", "y"]]

model = Prophet(growth="linear", daily_seasonality=False, n_changepoints=2)

model.fit(df)

future = model.make_future_dataframe(periods=0)

forecast = model.predict(future)

fig = model.plot(forecast)

_ = add_changepoints_to_plot(fig.gca(), model, forecast)
df = trend_df.rename({"Log10(Deaths)": "y"}, axis=1).loc[:, ["ds", "y"]]

model = Prophet(growth="linear", daily_seasonality=False, n_changepoints=2)

model.fit(df)

future = model.make_future_dataframe(periods=0)

forecast = model.predict(future)

fig = model.plot(forecast)

_ = add_changepoints_to_plot(fig.gca(), model, forecast)
df = trend_df.rename({"Log10(Recovered)": "y"}, axis=1).loc[:, ["ds", "y"]]

model = Prophet(growth="linear", daily_seasonality=False, n_changepoints=2)

model.fit(df)

future = model.make_future_dataframe(periods=0)

forecast = model.predict(future)

fig = model.plot(forecast)

_ = add_changepoints_to_plot(fig.gca(), model, forecast)
train_dataset = SIR.create_dataset(

    ncov_df, population_dict["Global"] - population_dict["China"], excluded_places=[("China", None)]

)

train_start_date, train_initials, train_Tend, train_df = train_dataset

pprint([train_start_date.strftime(time_format), train_initials, train_Tend])
train_df.tail()
line_plot(

    train_df.set_index("T").drop("x", axis=1),

    "Training data: y(T), z(T)", math_scale=False, ylabel=""

)
eg_r0, eg_rho = (2.5, 0.2)

eg_sigma = eg_rho / eg_r0

(eg_rho, eg_sigma)
%%time

eg_df = simulation(SIR, train_initials, step_n=300, rho=eg_rho, sigma=eg_sigma)

eg_df.tail()
line_plot(

    eg_df.set_index("t"),

    title=r"SIR: $R_0$={0} ($\rho$={1}, $\sigma$={2})".format(eg_r0, eg_rho, eg_sigma),

    ylabel="",

    h=1

)
# Set the example conditions

eg_tau = 1440

eg_start_date = ncov_df["Date"].min()

eg_total_population = 1000000

# Create dataset in the format of ncov_df

eg_ori_df = pd.DataFrame(

    {

        "Date": (eg_df["t"] * eg_tau).apply(lambda x: timedelta(minutes=x)) + eg_start_date,

        "Country": "Example",

        "Province": "Example"

    }

)

eg_ori_df["Infected"] = (eg_df["y"] * eg_total_population).astype(int)

eg_ori_df["Deaths"] = (eg_df["z"] * eg_total_population * 0.02).astype(int)

eg_ori_df["Recovered"] = (eg_df["z"] * eg_total_population * 0.98).astype(int)

eg_ori_df.tail()
# line_plot(eg_ori_df.set_index("Date")[data_cols], "Example data")
# %%time

# eg_sir_estimater = Estimater(SIR, eg_ori_df, eg_total_population, places=[("Example", "Example")])

# eg_sir_dict = eg_sir_estimater.run()
# eg_sir_estimater.compare_graph()
"""

eg_dict = {

    "Condition": {

        "tau": eg_tau, "rho": eg_rho, "sigma": eg_sigma,

        "R0": eg_r0, "score": 0, **SIR(rho=eg_rho, sigma=eg_sigma).calc_days_dict(eg_tau)

    },

    "Estimation": eg_sir_dict

}

df = pd.DataFrame.from_dict(eg_dict, orient="index")

df

"""

None
# eg_sir_estimater.predict_graph(step_n=500, name="Example area")
%%time

sir_estimater = Estimater(SIR, ncov_df, population_dict["Global"] - population_dict["China"], excluded_places=[("China", None)])

sir_dict = sir_estimater.run()
sir_estimater.history_df().head()
sir_estimater.history_graph()
sir_dict
sir_estimater.compare_graph()
sir_estimater.predict_graph(step_n=400, name="Total except China")
# %%time

# sird_estimater = Estimater(

#     SIRD, ncov_df, population_dict["Global"] - population_dict["China"], excluded_places=[("China", None)]

# )

# sird_dict = sird_estimater.run()
# sird_estimater.history_graph()
# sird_dict
# sird_estimater.compare_graph()
# sird_estimater.predict_graph(step_n=500, name="Total except China")
%%time

sirf_estimater = Estimater(

    SIRF, ncov_df, population_dict["Global"] - population_dict["China"], excluded_places=[("China", None)]

)

sirf_dict = sirf_estimater.run()
sirf_estimater.history_df().head()
sirf_estimater.history_graph()
sirf_dict
sirf_estimater.compare_graph()
sirf_estimater.predict_graph(step_n=500, name="Total except China")
_dict = {

    "SIR": sir_dict,

    # "SIR-D": sird_dict,

    "SIR-F": sirf_dict

}

model_param_df = pd.DataFrame.from_dict(_dict, orient="index")

model_param_df.fillna("-")
country_df = ncov_df.pivot_table(

    values="Infected", index="Date", columns="Country", aggfunc=sum

).fillna(0).astype(int)

country_df = country_df.drop("China", axis=1)
line_plot(

    country_df.T.nlargest(5, country_df.index.max()).T,

    "Infected in top 5 countries without China",

    math_scale=False

)
_, jp_df = create_target_df(ncov_df, population_dict["Japan"], places=[("Japan", None)])

jp_df.tail()
line_plot(jp_df.set_index("T")[data_cols], "Japan: without Susceptible", math_scale=False)
%%time

jp_sirf_estimater = Estimater(SIRF, ncov_df, population_dict["Japan"], places=[("Japan", None)])

jp_sirf_dict = jp_sirf_estimater.run()
jp_sirf_dict
# jp_sirf_estimater.history_graph()
jp_sirf_estimater.compare_graph()
jp_sirf_estimater.predict_graph(step_n=500, name="Japan")
sk_start_date, sk_df = create_target_df(ncov_df, population_dict["South Korea"], places=[("South Korea", None)])

sk_df.tail()
line_plot(sk_df.set_index("T")[data_cols], "South Korea: without Susceptible", math_scale=False)
sk_start_date + timedelta(minutes=40000)
%%time

sk_sirf_estimater = Estimater(SIRF, ncov_df, population_dict["South Korea"], places=[("South Korea", None)])

sk_sirf_dict = sk_sirf_estimater.run()
# sk_sirf_estimater.history_graph()
sk_sirf_dict
sk_sirf_estimater.compare_graph()
sk_sirf_estimater.predict_graph(step_n=500, name="South Korea")
_, it_df = create_target_df(ncov_df, population_dict["Italy"], places=[("Italy", None)])

it_df.tail()
line_plot(it_df.set_index("T")[data_cols], "Italy: without Susceptible", math_scale=False)
%%time

it_sirf_estimater = Estimater(SIRF, ncov_df, population_dict["Italy"], places=[("Italy", None)])

it_sirf_dict = it_sirf_estimater.run()
it_sirf_dict
# it_sirf_estimater.history_graph()
it_sirf_estimater.compare_graph()
it_sirf_estimater.predict_graph(step_n=500, name="Italy")
_, ir_df = create_target_df(ncov_df, population_dict["Iran"], places=[("Iran", None)])

ir_df.tail()
line_plot(ir_df.set_index("T")[data_cols], "Iran: without Susceptible", math_scale=True)
%%time

ir_sirf_estimater = Estimater(SIRF, ncov_df, population_dict["Iran"], places=[("Iran", None)])

ir_sirf_dict = ir_sirf_estimater.run()
ir_sirf_dict
# ir_sirf_estimater.history_graph()
ir_sirf_estimater.compare_graph()
ir_sirf_estimater.predict_graph(step_n=500, name="Iran")
_dict = {

    "Total except China": sirf_dict,

    "Japan": jp_sirf_dict,

    "South Korea": sk_sirf_dict,

    "Italy": it_sirf_dict,

    "Iran": ir_sirf_dict,

}

comp_param_df = pd.DataFrame.from_dict(_dict, orient="index")

comp_param_df
sirf_dict
country = "Except China"

total_population = population_dict["Global"] - population_dict["China"]

first_estimater = sirf_estimater

param_dict = first_estimater.study.best_params.copy()

param_dict
predicter = Predicter(country, first_estimater)

predicter.add(SIRF, days=30, **param_dict)

predicter.restore_graph("Susceptible")
predicter = Predicter(country, first_estimater)

predicter.add(SIRF, days=1000, **param_dict)

predicter.restore_graph()
changed_param_dict = param_dict.copy()

changed_param_dict["rho"] = param_dict["rho"] / 2

predicter = Predicter(country, first_estimater)

predicter.add(SIRF, days=1000, **changed_param_dict)

predicter.restore_graph()
changed_param_dict = param_dict.copy()

changed_param_dict["rho"] = param_dict["rho"] / 2

predicter = Predicter(country, first_estimater)

predicter.add(SIRF, days=30, **changed_param_dict)

predicter.restore_graph("Susceptible")
changed_param_dict = param_dict.copy()

changed_param_dict["sigma"] = param_dict["sigma"] * 2

changed_param_dict["kappa"] = param_dict["kappa"] / 2

predicter = Predicter(country, first_estimater)

predicter.add(SIRF, days=1000, **changed_param_dict)

predicter.restore_graph()
changed_param_dict = param_dict.copy()

changed_param_dict["n"] = total_population

changed_param_dict["v_per_day"] = 1000000

predicter = Predicter(country, first_estimater)

predicter.add(SIRFV, days=1000, **changed_param_dict)

predicter.restore_graph()