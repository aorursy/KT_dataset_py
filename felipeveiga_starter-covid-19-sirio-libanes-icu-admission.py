import pandas as pd 

import matplotlib.pyplot as plt 

plt.style.use('ggplot')



from IPython.display import display_html
import os

print(os.listdir("../"))
data = pd.read_excel("../input/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")



comorb_lst = [i for i in data.columns if "DISEASE" in i]

comorb_lst.extend(["HTN", "IMMUNOCOMPROMISED", "OTHER"])



demo_lst = [i for i in data.columns if "AGE_" in i]

demo_lst.append("GENDER")



vitalSigns_lst = data.iloc[:,193:-2].columns.tolist()



lab_lst = data.iloc[:,13:193].columns.tolist()
print(f"Number of Comorbities features: {len(comorb_lst)}") 

print(f"Number of Demographics features: {len(demo_lst)}") 

print(f"Number of Vital Signs features: {len(vitalSigns_lst)}") 

print(f"Number of Laboratory features: {len(lab_lst)}") 
# ID is a identification number for each patient.

print(f"Number of lines in the dataset: {len(data)}")

print(f"Number of inpatients: {len(data.PATIENT_VISIT_IDENTIFIER.unique())}")
print(data.WINDOW.unique())
data.groupby("PATIENT_VISIT_IDENTIFIER", as_index = False).agg({"ICU":(list), "WINDOW":list}).iloc[[13,14,15,41,0,2]]
aux = abs(data.groupby("PATIENT_VISIT_IDENTIFIER")["ICU"].sum()-5)

aux = aux.value_counts().reset_index()

aux.sort_values(by = "index", inplace = True)

aux.reset_index(drop = True, inplace = True)
tot_icu_inpatients = aux.ICU[0:5].sum()

y = aux.ICU[0:5].cumsum()/tot_icu_inpatients

plt.plot(y, marker = ".")



plt.ylabel

plt.xlabel("Window")

plt.yticks(round(y,2) )

plt.xticks([0,1,2,3,4], ["0-2", "2-4", "4-6", "6-12", "Above-12"])

plt.show()
missing_df = data.groupby("WINDOW").count()/249
missing_df[vitalSigns_lst]
df1_styler = missing_df[demo_lst].style.set_table_attributes("style='display:inline'").set_caption('Demographics')

df2_styler = missing_df[comorb_lst].style.set_table_attributes("style='display:inline'").set_caption('Comorbities')



display_html(df1_styler._repr_html_()+df2_styler._repr_html_(), raw=True)
data[data["PATIENT_VISIT_IDENTIFIER"] == 1]
data[data["PATIENT_VISIT_IDENTIFIER"] == 0]