import pandas as pd
import numpy as np
sal=pd.read_csv("../input/Salaries.csv")
sal.head()
sal.info()
# sal["BasePay"].mean()
# sal["OvertimePay"].max()
sal[sal["EmployeeName"]=="JOSEPH DRISCOLL"]["JobTitle"]
sal[sal["EmployeeName"]=="JOSEPH DRISCOLL"]["TotalPayBenefits"]
# sal[sal["TotalPayBenefits"].max()]
sal[sal["TotalPayBenefits"]==sal["TotalPayBenefits"].max()]
sal[sal["TotalPayBenefits"]==sal["TotalPayBenefits"].min()]
# sal.groupby("Year")["BasePay"].mean()
sal["JobTitle"].nunique()
sal["JobTitle"].value_counts().head(5)
sum(sal[sal["Year"]==2013]["JobTitle"].value_counts()==1)
def check(job):
    if "chief" in job.lower():
        return True
    else:
        return False
sum(sal["JobTitle"].apply(lambda x: check(x)))
sal["JT_len"]=sal["JobTitle"].apply(len)
sal.head(2)
sal[["JT_len","TotalPayBenefits"]].corr()