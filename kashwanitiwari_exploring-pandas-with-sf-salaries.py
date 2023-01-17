
import pandas as pd
salaries = pd.read_csv('../input/Salaries.csv')
salaries.head()
salaries.info()
salaries.describe().transpose()
salaries["TotalPay"].mean()
salaries["TotalPayBenefits"].max()
salaries[salaries["EmployeeName"]=="PATRICIA JACKSON"]["JobTitle"]
salaries[salaries["EmployeeName"]=="PATRICIA JACKSON"]["TotalPayBenefits"]
salaries[salaries["TotalPayBenefits"]==salaries.min()["TotalPayBenefits"]]
salaries[salaries["TotalPayBenefits"]==salaries.max()["TotalPayBenefits"]]
salaries.groupby('Year').mean()['TotalPay']
salaries["JobTitle"].nunique()
salaries["JobTitle"].value_counts().head()