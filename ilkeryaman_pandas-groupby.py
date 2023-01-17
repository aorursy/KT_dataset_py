import pandas as pd # data processing
dataset = {
    "Department": ["IT", "HR", "Finance", "Finance", "IT", "HR"],
    "Employee": ["Ilker", "Erhan", "Ali", "Serdar", "Esra", "Recep"],
    "Salary": [3000, 3500, 2500, 4500, 4000, 2000]
}
df = pd.DataFrame(dataset)

df
dep_group = df.groupby("Department") # Group by Department

dep_group
dep_group.sum() # Get total salaries for departments.
dep_group.mean() # Get mean salaries for departments.
dep_group.mean().loc["IT"] # Get mean salary for IT department
int(dep_group.mean().loc["IT"]) # Get real value of mean salary for IT department
dep_group.count() # Count of employee and salary for each department
dep_group.max() # Maximum salary owner for each department.
dep_group.min() # Minimum salary owner for each department.
dep_group.min()["Salary"][0] # Get minimum salary for first department in dataframe.
dep_group.min()["Salary"]["Finance"] # Get minimum salary for Finance department.