import pandas as pd # Standard way to import pandas
# Create data for table/dataframe
header = ["Name", "Attendance", "CGPA"]
rows = [["Mihir", 93, 8.9],
       ["Suriya", 90, 9.1],
       ["Kumar", 96, 8.2],
       ["Paul", 85, 7.9],
       ["Biju", 97, 9.5],
       ["Dipak", 82, 8.4]]
df = pd.DataFrame(rows, columns=header)
df
df.head() # display top 5 rows
df.columns
df.to_csv("attendance.csv", header=True, index=False) # Write the dataframe to CSV file
df = pd.read_csv("attendance.csv")
df
print(df.Attendance.mean())
# OR
print(df["Attendance"].mean())
df.CGPA.std() # standard deviation
df.sort_values(["Attendance"]) # Sort ASC based on `Attendance` column
df.sort_values(["Attendance"]).reset_index(drop=True) # Reset index
df.values # get the numpy array from the dataframe
# NOTE: dtype of array is `object`
df_copy = df.copy()
df_copy.Attendance = df_copy.Attendance.astype(float) # change dtype of pandas dataframe
df_copy
print(df.CGPA.values)
ind = df.CGPA.values.argmax() # Get index with maximum CGPA
print(ind)
df.loc[ind]                   # Get row with index `ind`
df.CGPA.plot()
df.Attendance.plot(c='r')
df[["Attendance", "CGPA"]].plot()
df["Age"] = [45, 23, 43, 34, 29, 40] # Add new column
df
df["useless_column"] = [1, 2, 3, 4] # Try to add column with different number of rows
df.shape
df.loc[df.shape[0]] = ["Anand", 60, 9.0, 42] # Add new entry
df
print(df.shape)
# OR
print(df.values.shape)
# IMPORTANT - `loc` function
# .loc can take one or two indices.
# If both are given, first is for rows, second is for columns
df_copy = df.copy()

df_copy.loc[df_copy.Attendance < 75, "Attendance"] = 75.0

print(df)
print()
print(df_copy)
df.loc[df.shape[0]] = ["Kamal", 57, 7.0, 39] # Add new entry
df_copy = df.copy()
print("Attendance < 75% : ", (df_copy.Attendance < 75).values)
print("CGPA > 8.5       : ", (df.CGPA > 8.5).values)

indices = (df_copy.Attendance < 75).values * (df.CGPA > 8.5).values # students with CGPA > 8.5 and attendance < 75%
df_copy.loc[indices, "Attendance"] = 75.0
df_copy
df
df.loc[3: 5, ["CGPA", "Name"]] # NOTE: 3: 5 - both 3 and 5 are inclusive
df.loc[3: 4] # if only one index is provided, all columns are included, i.e. equivalent to .loc[3: 4, :]

print(df.Attendance.values)
df.Attendance.hist(figsize=(10, 6))