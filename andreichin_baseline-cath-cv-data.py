# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
import seaborn as sns

cath_data = pd.read_excel(open("/kaggle/input/baseline-cath-cv-data-updated/baseline_iv_cv_imaging.xls", "rb"), sheet_name="Sheet1", header=None)
print(cath_data.head())
# Need to remove the first row
cath_data.columns = cath_data.iloc[0]
cath_data = cath_data[1:]
cath_data.head()
cath_data.describe()
cv_data = cath_data.copy()
cv_data.rename(columns={"Date": "date", 
                          "Time": "time",
                         "Exam type": "exam_type",
                         "Gauge size required per protocol": "gauge_size_required_per_protocol",
                         "Gauge of IV attempted": "gauge_of_iv_attempted",
                         "Gauge of IV successfully inserted": "gauge_of_successfully_inserted",
                         "Number of insertion attempts": "number_of_insertion_attempts",
                         "IV team / expert called \n(Y / N)": "iv_team_expert_called",
                         "Minutes waiting on IV team / expert": "minutes_waiting_on_iv_team_expert",
                         "PICC line placed\n(Y / N)": "picc_line_placed",
                         "Location": "location",
                         "Additional comments": "additional_comments",
                         "Protocol flow rate (mL / sec)": "protocol_flow_rate",
                         "Selected flow rate based on patient factors (mL / sec)": "selected_flow_rate_based_on_patient_factors",
                         "Flow rate achieved\n(mL / sec)": "flow_rate_achieved"},inplace=True)
cv_data.describe()
cv_data["gauge_size_required_per_protocol"].value_counts(dropna=False)
sns.set(font_scale=1.4)
cv_data["gauge_size_required_per_protocol"].value_counts().plot(kind="bar", figsize=(7,6), rot=0, color=("pink", "blue"))
plt.xlabel("IV Gauge Size Required Per Protocol", labelpad=14)
plt.ylabel("Number", labelpad=14)
plt.title("Gauge Size Required Per Protocol", y=1.02)
cv_data.exam_type.unique()
ct_data = cv_data[(cv_data.exam_type == "CTA") | 
                  (cv_data.exam_type == "AFIB") | 
                  (cv_data.exam_type == "TAVR") | 
                  (cv_data.exam_type == "CT LIVER") | 
                  (cv_data.exam_type == "THROMBUS")]
ct_data.describe()
ct_data.exam_type.value_counts()
sns.set(font_scale=1.4)
ct_data.exam_type.value_counts().plot(kind="bar", figsize=(7,6), rot=0, color=("Red", "Orange", "Blue", "Purple", "Green"))
plt.xlabel("CT Protocols", labelpad=14)
plt.xticks(rotation=90)
plt.ylabel("Number of Protocols", labelpad=14)
plt.title("Distribution of CT Protocols ")
cmr_data = cv_data[(cv_data.exam_type == "CMR") | (cv_data.exam_type == "MRA")]
cmr_data.describe()
cmr_data.exam_type.value_counts()
sns.set(font_scale=1.4)
cmr_data.exam_type.value_counts().plot(kind="bar", figsize=(7,6), rot=0, color=("blue", "red"))
plt.xlabel("CMR Protocol", labelpad=14)
plt.ylabel("Number of CMR Protocols", labelpad=14)
plt.title("Distribution of CMR Protocols")
ct_data.gauge_size_required_per_protocol.value_counts()
ct_data.gauge_size_required_per_protocol.value_counts().plot(kind="bar")
ct_gauge_data = ct_data[["exam_type", "gauge_size_required_per_protocol"]]
ct_gauge_data.head()
ct_gauge_data.groupby(["exam_type", "gauge_size_required_per_protocol"]).size()
ct_gauge_data.groupby(["exam_type", "gauge_size_required_per_protocol"]).size().plot(kind="bar", figsize=(7,6), rot=0, color=("blue", "red", "orange", "purple", "yellow", "green"))
sns.set(font_scale=1.4)
#cmr_data.exam_type.value_counts().plot(kind="bar", figsize=(7,6), rot=0, color=("blue", "red"))
plt.xlabel("CT Protocol", labelpad=14)
plt.xticks(rotation=90)
plt.ylabel("Number", labelpad=14)
plt.title("Distribution of Gauge Size Required Per Protocol")
cmr_data.groupby(["exam_type", "gauge_size_required_per_protocol"]).size().plot(kind="bar")
cv_data.groupby(["gauge_size_required_per_protocol", "gauge_of_iv_attempted", "gauge_of_successfully_inserted", "number_of_insertion_attempts"]).size()
cv_data["number_of_insertion_attempts"].mean()
import squarify # pip install squarify (algorithm for treemap)

# Change color
'''
squarify.plot(sizes=[13,22,35,5], label=["group A", "group B", "group C", "group D"], color=["red","green","blue", "grey"], alpha=.4 )
plt.axis('off')
plt.show()
'''

x = cv_data.groupby(['gauge_of_iv_attempted', 'number_of_insertion_attempts']).size()
x_group = cv_data.groupby(['gauge_of_iv_attempted', 'number_of_insertion_attempts'])
x
is_20 = cv_data["gauge_of_iv_attempted"] == 20
gauge_20 = cv_data[is_20]
gauge_20.head()
gauge_20.groupby(['gauge_of_iv_attempted', 'number_of_insertion_attempts']).size()
xyz = gauge_20.groupby(['gauge_of_iv_attempted', 'number_of_insertion_attempts']).size()
labels = ["5", "4", "3", "2", "1"]
#squarify.plot(sizes=)
# If you have a data frame?
#df = pd.DataFrame({'nb_people':[8,3,4,2], 'group':["group A", "group B", "group C", "group D"] })
xyz = gauge_20.groupby(['gauge_of_iv_attempted', 'number_of_insertion_attempts']).size()
labels = ["1", "2", "3", "4", "5"]
squarify.plot(sizes=xyz, label=labels, alpha=.8 )
plt.axis('off')
plt.show() 

cv_data.groupby(["gauge_size_required_per_protocol", "gauge_of_successfully_inserted"]).size().plot(kind="pie")
z=np.random.rand(40)
plt.scatter(cv_data['gauge_of_iv_attempted'], cv_data['number_of_insertion_attempts'], s = z*1000)
plt.show()
is_22 = cv_data["gauge_of_iv_attempted"] == 22
gauge_22 = cv_data[is_22]
gauge_22.head()
abc = gauge_22.groupby(['gauge_of_iv_attempted', 'number_of_insertion_attempts']).size()
labels = ["1", "2", "3"]
squarify.plot(sizes=abc, label=labels, alpha=.8 )
plt.axis('off')
plt.show() 
cath_success = cv_data.groupby(["gauge_size_required_per_protocol","gauge_of_successfully_inserted"]).size()
cath_success
cath_success.plot(kind="bar")
cath_attempt = cv_data.groupby(["gauge_of_iv_attempted","number_of_insertion_attempts"]).size()
cath_attempt
cath_attempt.plot(kind="bar")
cv_data["iv_team_expert_called"].describe()
# filtering df for unsuccessful attempts in 20 gauge IVs which resulted to resorting in lower gauge IV
per_protocol_20 = cv_data[(cv_data.gauge_size_required_per_protocol==20) & (cv_data.gauge_of_iv_attempted==22) & (cv_data.gauge_of_successfully_inserted==22)] 
per_protocol_20
per_protocol_20.groupby(["gauge_of_iv_attempted", "gauge_of_successfully_inserted"]).size()
cv_data["iv_team_expert_called"].describe()

cv_data["iv_team_expert_called"].value_counts().plot(kind="bar")
protocol_fr = cv_data["protocol_flow_rate"]
protocol_fr.value_counts().plot(kind="bar")
ct_protocol_fr = ct_data["protocol_flow_rate"]
ct_protocol_fr.value_counts().plot(kind="bar")
ct_protocol_fr.describe()
selected_fr = ct_data["selected_flow_rate_based_on_patient_factors"]
selected_fr.value_counts().plot(kind="bar")
selected_fr.value_counts()
fr_achieved = ct_data["flow_rate_achieved"]
fr_achieved.value_counts().plot(kind="bar")
dif_cath_data = pd.read_excel(open("/kaggle/input/cv-diffusics-data/diffusics_iv_cv_imaging.xls", "rb"), sheet_name="Sheet1", header=None)
dif_cath_data.head()
# Need to remove the first row

dif_cath_data_header = dif_cath_data.iloc[0]
dif_cath_data = dif_cath_data[1:]
dif_cath_data.columns = dif_cath_data_header 
dif_cath_data.head()
dif_cath_data.describe()
dif_cath_data_copy = dif_cath_data.copy()
dif_cath_data_copy.rename(columns={"Date": "date", 
                          "Time": "time",
                         "Exam type": "exam_type",
                         "Gauge size required per protocol": "gauge_size_required_per_protocol",
                         "Gauge of IV attempted": "gauge_of_iv_attempted",
                         "Gauge of IV successfully inserted": "gauge_of_successfully_inserted",
                         "Number of insertion attempts": "number_of_insertion_attempts",
                         "IV team / expert called \n(Y / N)": "iv_team_expert_called",
                         "Minutes waiting on IV team / expert": "minutes_waiting_on_iv_team_expert",
                         "PICC line placed\n(Y / N)": "picc_line_placed",
                         "Location": "location",
                         "Additional comments": "additional_comments",
                         "Protocol flow rate (mL / sec)": "protocol_flow_rate",
                         "Selected flow rate based on patient factors (mL / sec)": "selected_flow_rate_based_on_patient_factors",
                         "Flow rate achieved\n(mL / sec)": "flow_rate_achieved"},inplace=True)
dif_cath_data_copy.describe()
dif_cath_data_copy["gauge_size_required_per_protocol"].value_counts(dropna=False)
dif_cath_data_copy.exam_type.unique()
# remove whitespace from 'CTA '
#df1['employee_id'] = df1['employee_id'].str.strip()
dif_cath_data_copy["exam_type"] = dif_cath_data_copy["exam_type"].str.strip()
# Filter CT data out of the DF
ct_dc_data = dif_cath_data_copy[(dif_cath_data_copy.exam_type == "CTA") | 
                  (dif_cath_data_copy.exam_type == "AFIB") | 
                  (dif_cath_data_copy.exam_type == "TAVR") | 
                  (dif_cath_data_copy.exam_type == "AP")]
ct_dc_data.describe()
dif_cath_data_copy.groupby(["gauge_size_required_per_protocol", "gauge_of_iv_attempted", "gauge_of_successfully_inserted", "number_of_insertion_attempts"]).size()
# cv_data.groupby(["gauge_size_required_per_protocol", "gauge_of_iv_attempted", "gauge_of_successfully_inserted", "number_of_insertion_attempts"]).size()
ct_dc_data.groupby(["gauge_size_required_per_protocol", "gauge_of_iv_attempted", "gauge_of_successfully_inserted", "number_of_insertion_attempts"]).size()
jelco = dif_cath_data_copy[(dif_cath_data_copy.additional_comments=="JELCO")]
jelco.describe()
jelco
dif_cath_data_copy["iv_team_expert_called"].describe()
df1 = pd.DataFrame(cv_data, columns=["flow_rate_achieved"]) #jelco
df2 = pd.DataFrame(dif_cath_data_copy, columns=["flow_rate_achieved"]) #bd

df1_average = ct_data["flow_rate_achieved"].mean()
df2_average = ct_dc_data["flow_rate_achieved"].mean()

df1_average
#df2_average
no_jelco_dif_data =  ct_dc_data[(ct_dc_data.additional_comments!="JELCO")]
no_jelco_dif_data.describe()
no_jelco_dif_data["number_of_insertion_attempts"].mean()
df2_average = no_jelco_dif_data["flow_rate_achieved"].mean()
df2_average
ct_df_fr = ct_dc_data["flow_rate_achieved"]
ct_df_fr.value_counts().plot(kind="bar")
no_jelco_dif_data["flow_rate_achieved"].value_counts().plot(kind="bar")
dif_fra_col = no_jelco_dif_data["flow_rate_achieved"]
dif_fra_col
# We are going to compare the Jelco 20g IV and Diffusics 22g IV flow rate.
jel_ct_20g_data_fr = ct_data[(ct_data.gauge_of_successfully_inserted==20)] # Filter Jelco data to get gauge of successfully inserted IV in 20g
jel_ct_20g_data_fr.describe()
jel_ct_22g_data_fr = ct_data[(ct_data.gauge_of_successfully_inserted==22)]
jel_ct_22g_data_fr.describe()
jel_ct_20g_data_fr['number_of_insertion_attempts'].value_counts()
jel_ct_22g_data_fr["flow_rate_achieved"].mean()
dc_ct_22g_data_fr = no_jelco_dif_data[(no_jelco_dif_data.gauge_of_successfully_inserted==22)] # Filter Diffusics data to get gauge of successfully inserted IV in 22g
dc_ct_22g_data_fr.describe()
jel_ct_20g_data_fr["flow_rate_achieved"].mean()
dc_ct_22g_data_fr["flow_rate_achieved"].mean()
jel_ct_22g_data_fr = ct_data[(ct_data.gauge_of_successfully_inserted==22)]
jel_ct_22g_data_fr.describe()
jel_ct_22g_data_fr["flow_rate_achieved"].mean()
dc_ct_20g_data_fr = no_jelco_dif_data[(no_jelco_dif_data.gauge_of_successfully_inserted==20)] # Filter Diffusics data to get gauge of successfully inserted IV in 20g
dc_ct_20g_data_fr.describe()
dc_ct_20g_data_fr["flow_rate_achieved"].mean() # average for Diffusics catheter
no_jelco_dif_data["number_of_insertion_attempts"].mean() # average for insertion attempts for diffusics catheter
jelco_ins_att = ct_data["number_of_insertion_attempts"]
jelco_ins_att.mean() # average for insertion attempts for jelco catheterb
no_jelco_dif_data["additional_comments"].value_counts()

ct_data["additional_comments"].value_counts() # Jelco extravasation
no_jelco_dif_data["number_of_insertion_attempts"].mean() # df mean for number of attempts
noia = pd.DataFrame({'Catheters':['Jelco', 'Diffusics'], 'Attempts':[1.29, 1.13]})
ax = noia.plot.bar(x='Catheters', y="Attempts", rot=0, color=['#5cb85c','#5bc0de'], legend=False, title="Insertion Attempts")
ct_ir_20g = pd.DataFrame({'Catheters':['Jelco', 'Diffusics'], 'Infusion Rates':[5.02, 5.01]})
ax = ct_ir_20g.plot.bar(x='Catheters', y="Infusion Rates", rot=0, color=['#5cb85c','#5bc0de'], legend=False, title="Infusion Rate")
extravasation = pd.DataFrame({'Catheters':['Jelco', 'Diffusics'], 'Extravasation':[2, 1]})
ax = extravasation.plot.bar(x='Catheters', y="Extravasation", rot=0, color=['#5cb85c','#5bc0de'], legend=False, title="Extravasation")
jc_vs_df_20_ir = pd.DataFrame({'Catheters':['Jelco', 'Diffusics'], 'Infusion Rate':[5.02, 5]})
ax = jc_vs_df_20_ir.plot.bar(x='Catheters', y="Infusion Rate", rot=0, color=['#5cb85c','#5bc0de'], legend=False, title="Infusion Rate 20G IV")
cv_data["iv_team_expert_called"].value_counts().plot(kind="bar")
cv_data["iv_team_expert_called"].value_counts().plot(kind="bar")
cv_data["iv_team_expert_called"].value_counts()
dif_cath_data_copy["iv_team_expert_called"].value_counts()
iv_expert_called = pd.DataFrame({'Catheters':['Jelco', 'Diffusics'], 'IV Expert Called':[6, 0]})
ax = iv_expert_called.plot.bar(x='Catheters', y="IV Expert Called", rot=0, color=['#5cb85c','#5bc0de'], legend=False, title="IV Expert Called")
#jc_vs_df_20_ir = pd.DataFrame({'Catheters':['Jelco', 'Diffusics'], 'Infusion Rate':[5.02, 5]})
#ax = jc_vs_df_20_ir.plot.bar(x='Catheters', y="Infusion Rate", rot=0, color=['#5cb85c','#5bc0de'], legend=False, title="Infusion Rate 20G IV")
# There were instances that nurses used Jelco during the Diffusics trial stage.  Those data were removed to focus solely on Diffusics catheters.
#5.024590163934426 Jelco 20g
#4.777777777777778 BD 22g
jelco_20_vs_df22 = pd.DataFrame({'Catheters':['Jelco', 'Diffusics'], 'Jelco 20g vs Diffusics 22g':[5.02, 4.78]})
ax = jelco_20_vs_df22.plot.bar(x='Catheters', y="Jelco 20g vs Diffusics 22g", rot=0, color=['#5cb85c','#5bc0de'], legend=False, title="Jelco 20g vs Diffusics 22g")
## combining bar charts for jelco and diffusics insertion attempts

jelco_vs_diffusics_ir = pd.DataFrame({
    "20g":[5.02, 5],
    "22g":[4.52, 4.78]
    }, 
    index=["Jelco", "Diffusics"]
)
jelco_vs_diffusics_ir.plot(kind="bar")
plt.title("Jelco vs Diffusics Rate of Infusion")
plt.xlabel("Catheters")
plt.ylabel("Rate of Infusion")
plt.legend(loc="center")

df_cath_attempt = dif_cath_data_copy.groupby(["gauge_of_iv_attempted","number_of_insertion_attempts"]).size()
df_cath_attempt
df_cath_attempt.plot(kind="bar")