import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from numpy import where as IF
claims = pd.read_csv("../input/eda-on-insurance-claim-dataset/claims.csv")

demo = pd.read_csv("../input/demographics/demo.csv")

demo.head(2)
claims.head(2)
comb_df = pd.merge(right = claims,

                   left = demo, 

                   right_on = "customer_id", 

                   left_on = "CUST_ID",

                   how = "outer"

                  )

comb_df.drop(columns = ["customer_id"], inplace = True)

comb_df.head(2)
comb_df.head()
comb_df["DateOfBirth"] = pd.to_datetime(comb_df.DateOfBirth, format = "%d-%b-%y")

comb_df.loc[(comb_df.DateOfBirth.dt.year > 2020),"DateOfBirth"]=comb_df[comb_df.DateOfBirth.dt.year > 2020]["DateOfBirth"].apply(lambda x: x - pd.DateOffset(years=100))

comb_df["claim_date"] = pd.to_datetime(comb_df.claim_date, format = "%m/%d/%Y")

comb_df["Contact"] = pd.to_numeric(comb_df.Contact.str.replace("-",""),downcast='float')

comb_df["claim_amount"] = pd.to_numeric(comb_df.claim_amount.str.replace("$",""),downcast='float')

comb_df.head(2)


comb_df["flag"] = IF(comb_df.police_report == "No", 0 ,

                    IF(comb_df.police_report == "Yes", 1, np.nan))

comb_df.drop(columns = ["police_report"], inplace = True)
comb_df = comb_df.groupby('CUST_ID').first().reset_index(drop = True)
comb_df.head()
comb_df["incident_cause"].isna().sum()
cat_col = ["gender","State","Segment","incident_cause","claim_area","claim_type","fraudulent","flag"]

con_col = ["claim_amount"]
for col in cat_col:

    comb_df[col] = comb_df[col].fillna(comb_df[col].mode()[0])

comb_df[con_col] = comb_df[con_col].fillna(comb_df[con_col].mean())

comb_df.head()
# comb_df["incident_cause"].fillna(0, inplace =True)

comb_df["State"].isna().sum()

comb_df["Age"] = round((comb_df.claim_date - comb_df.DateOfBirth).apply(lambda x: x.days)/365.25, 2)
comb_df["Age_grp"] = IF(comb_df.Age < 18, "Children",

                        IF(comb_df.Age < 30, "Youth",

                         IF(comb_df.Age < 60, "Adult",

                          IF(comb_df.Age < 100, "Senior", "NaN"

                           

                          )

                         )

                        )

                       )

comb_df["Age_grp"] = comb_df["Age_grp"].fillna(comb_df["Age_grp"].mode())

comb_df.groupby(by = "Age_grp").count()

# comb_df.head()
comb_df.groupby(by = "Segment")[["claim_amount"]].mean()
comb_df.loc[comb_df.claim_date < "2018-09-10",:].groupby("incident_cause")["claim_amount"].sum().add_prefix("total_")
comb_df.loc[(comb_df.incident_cause.str.lower().str.contains("driver") 

             & ((comb_df.State == "TX") | (comb_df.State == "DE") | (comb_df.State == "AK"))),:].groupby(by = "State")["State"].count()
f1 = comb_df.groupby(by = ["gender","Segment"])["claim_amount"].sum().reset_index()

f1.head()
res = f1.pivot(index = "Segment", columns = "gender", values = "claim_amount")

res
res.T.plot(kind = "pie", subplots = True, legend = False, figsize = (15,8))

plt.show()
f2 = comb_df.loc[(comb_df.incident_cause.str.lower().str.contains("driver"))].groupby(by = "gender")[["gender"]].count().add_prefix("CountOf_").reset_index()

f2
sns.barplot(x = "gender", y = "CountOf_gender", data = f2 )

plt.show()
comb_df.head()


comb_df.groupby(by = "Age_grp")[["fraudulent"]].count()
comb_df[(comb_df.Age_grp == np.nan)]
val = comb_df['Age_grp'].mode()[0]

print(val)

comb_df.loc[:,"Age_grp"] = comb_df.loc[:,'Age_grp'].fillna(value = val)

comb_df[(comb_df.Age_grp == "nan")]
comb_df['Age_grp'].mode()[0]