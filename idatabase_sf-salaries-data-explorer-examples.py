import pandas as pd
sal = pd.read_csv('../input/Salaries.csv',dtype={"Overtimdtype": object})
sal.head()
sal.info()
pd.to_numeric(sal['BasePay'], errors='coerce').dropna().mean()
pd.to_numeric(sal['OvertimePay'], errors='coerce').dropna().max()
sal.loc[sal['TotalPayBenefits'].idxmax()]['EmployeeName']
sal.loc[sal['TotalPayBenefits'].idxmin()]['EmployeeName']
sal.groupby('Year').mean()['TotalPayBenefits']
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head(1)
sum(sal['JobTitle'].apply(lambda x: 'chief' in x.lower()))