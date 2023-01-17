import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import os
print(os.listdir("../input"))
df_inspection = pd.read_csv("../input/inspections.csv")
df_violations = pd.read_csv("../input/violations.csv")
df_inspection.head()
#Let us view the facility type inspected:
df_inspection["facility_type"] = df_inspection["pe_description"].str.split("(").str[0]  #df_inspection["pe_description"].str.split("(").str.get(0)
print(df_inspection["facility_type"].value_counts(normalize=True))
sns.countplot(x = df_inspection["facility_type"])
plt.show()

#Let us view the 'Active' facility type inspected:
print("Status of Active facilities:")
print(df_inspection[df_inspection["program_status"]=="ACTIVE"]["facility_type"].value_counts(normalize=True))
sns.countplot(x = df_inspection[df_inspection["program_status"]=="ACTIVE"]["facility_type"])
plt.show()
#Let us analyse inspection counts over the years

#Create columns to analyse yearly inspection counts
df_inspection["inspection_month_num"] = pd.to_datetime(df_inspection["activity_date"]).dt.month
df_inspection["inspection_month"] = pd.to_datetime(df_inspection["activity_date"]).dt.strftime('%b')  
df_inspection["inspection_year"]  = pd.to_datetime(df_inspection["activity_date"]).dt.year

df_Inspections_Count =  df_inspection.groupby(["inspection_year","inspection_month_num"])["inspection_month"].count().reset_index().rename(columns={"inspection_month":"count"})

lst_Years = df_Inspections_Count["inspection_year"].unique().tolist()
for i in range(len(lst_Years)):
    print ("Year: {0}".format(lst_Years[i]))
    sns.barplot(x=df_Inspections_Count[df_Inspections_Count["inspection_year"]==lst_Years[i]]["inspection_month_num"], y=df_Inspections_Count[df_Inspections_Count["inspection_year"]==lst_Years[i]]["count"], palette="Blues_d")
    plt.show()
#Let us analyse inspection over the years by 'Faciity Type'
df_Inspections_Count =  df_inspection.groupby(["inspection_year","inspection_month_num","facility_type"])["inspection_month"].count().reset_index().rename(columns={"inspection_month":"count"})

lst_Years = df_Inspections_Count["inspection_year"].unique().tolist()
for i in range(len(lst_Years)):
    print ("Year: {0}".format(lst_Years[i]))
    sns.barplot(x=df_Inspections_Count[df_Inspections_Count["inspection_year"]==lst_Years[i]]["inspection_month_num"], y=df_Inspections_Count[df_Inspections_Count["inspection_year"]==lst_Years[i]]["count"],hue=df_Inspections_Count["facility_type"], palette="Blues_d")
    plt.legend(bbox_to_anchor=(1.01,1), loc=2)
    plt.show()
print ("Top 3 cities with the most inspections:\n{0}".format(df_inspection.groupby("facility_city")["facility_city"].count().sort_values(ascending=False).head(10)))
plt.figure(figsize=(8,35))
sns.countplot(y=df_inspection["facility_city"])
df_inspection[df_inspection["facility_city"].isin(["LOS ANGELES","GLENDALE","TORRANCE","BURBANK","SANTA MONICA","NORTH HOLLYWOOD","VAN NUYS","WHITTIER","POMONA","LANCASTER"])].groupby(["facility_city","facility_type"]).count().sort_values(by=["facility_city","facility_type"],ascending=True).iloc[:,:1].rename(columns={"activity_date":"Total" })
df_inspection.groupby("grade")["grade"].count()
df_inspection[df_inspection["grade"]==" "]="Unknown"
df_inspection.groupby("grade")["grade"].count()
#There are different kinds of inspection. 
#Let us understand if owners (of each graded entity they own) requested for an inspection or was it a routine inspection by the county

df_Inspections_Count = df_inspection.groupby(["inspection_year","service_description","grade"],as_index=False)["facility_city"].count().rename(columns={"facility_city":"Total"})

lst_Years = df_Inspections_Count["inspection_year"].unique().tolist()
for i in range(len(lst_Years)):
    print ("Year: {0}".format(lst_Years[i]))
    plt.figure(figsize=(10,4))
    sns.barplot(x=df_Inspections_Count[df_Inspections_Count["inspection_year"]==lst_Years[i]]["service_description"], y=df_Inspections_Count[df_Inspections_Count["inspection_year"]==lst_Years[i]]["Total"],hue=df_Inspections_Count["grade"], palette="Blues_d",  saturation=.55)
    plt.legend(bbox_to_anchor=(1.01,1), loc=2)
    plt.show()
#Let us analyse the violations
df_violations.head()
#We will create a DataFrame that holds columns from df_inspections and df_violations
df_inspections_violations = df_inspection[["serial_number", "facility_city","facility_state","grade","score","inspection_month_num","inspection_year", "facility_type","activity_date","record_id"]]

#Create a new DataFrame
df_inspections_violations = df_inspections_violations.merge(df_violations, how="inner", on="serial_number")
#Look at the merged DataFrame
df_inspections_violations.head()
#What kinds of violations do we have?
df_inspections_violations.groupby("violation_status")["violation_status"].count()
#For violation relating to "Out of Compliance", what are the violations codes?
df_inspections_violations[df_inspections_violations["violation_status"] =="OUT OF COMPLIANCE"].groupby("violation_code")["violation_code"].count().sort_values(ascending=False)
#Let us see the corresponding reasons for top 21 'violation' codes from above
lstviolation_codes= ["F044", "F033", "F035", "F040", "F036", "F037", "F043", "F007", "F030", "F039", "F014", "F006", "F023",     
                      "F034", "F052", "F029", "F027", "F042", "F046", "F025", "F038"]

pd.set_option('max_colwidth',100) #Setting the width of the column as default width does not display the whole text

df_reasons= df_inspections_violations[df_inspections_violations["violation_code"].isin (lstviolation_codes)].groupby(df_inspections_violations["violation_description"])[["violation_code","violation_description" ]].head(1)
df_reasons
#We remove the violation code from the description column
df_reasons["violation_description"] = df_reasons["violation_description"].str.split(".").str[1]

#Let us create a word cloud of the violation reasons and see 
violations_wc = WordCloud(    width=600, 
                              height=400, 
                              margin=1,
                              prefer_horizontal=.6, 
                              scale=2, 
                              max_words=200, 
                              background_color='black',
                              stopwords=STOPWORDS,
                              min_font_size=10,
                              random_state=101
                        ).generate(str(df_reasons["violation_description"]))

plt.figure(figsize=(12,12)),
plt.imshow(violations_wc)
plt.axis('off')
plt.show()
df_inspections_violations[df_inspections_violations["inspection_year"]==2016].groupby(["activity_date"])["inspection_year"].count()
#To see how many violations in the month of inspection

df_inspections_violations["activity_date"] = pd.to_datetime(df_inspections_violations["activity_date"])

#Year: 2015
df_yr2015 = df_inspections_violations[df_inspections_violations["inspection_year"]==2015].groupby(["activity_date"])["inspection_year"].count().to_frame()
df_yr2015.plot()
plt.show()

#Year: 2016
df_yr2016 = df_inspections_violations[df_inspections_violations["inspection_year"]==2016].groupby(["activity_date"])["inspection_year"].count().to_frame()
df_yr2016.plot()
plt.show()

#Year: 2017
df_yr2017 = df_inspections_violations[df_inspections_violations["inspection_year"]==2017].groupby(["activity_date"])["inspection_year"].count().to_frame()
df_yr2017.plot()
plt.show()
