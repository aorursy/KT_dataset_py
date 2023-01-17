import pandas as pd
dictionary = {"NAME": ["Ahmet", "Mehmet", "Osman", "Hatice", "Ayşe", "Fatıma"],

              "AGE": [15, 16, 17, 33, 45, 66],

              "SALARY": [1000, 1500, 2500, 3000, 3500, 4000]}



dataFrame = pd.DataFrame(dictionary)



print("dataFrame:\n{0}\n".format(dataFrame))

print("dataFrame.head(): [First 5 rows]\n{0}\n".format(dataFrame.head()))

print("dataFrame.tail(): [Last 5 rows]\n{0}\n".format(dataFrame.tail()))

print("dataFrame.info():")

print(dataFrame.info(), "\n")

print("dataFrame.columns:\n{0}\n".format(dataFrame.columns))

print("dataFrame.dtypes:\n{0}\n".format(dataFrame.dtypes))

print("dataFrame.describe():\n{0}\n".format(dataFrame.describe()))
print("dataFrame[\"AGE\"]:\n{0}\n".format(dataFrame["AGE"]))

print("dataFrame.NAME:\n{0}\n".format(dataFrame.NAME))
dataFrame["New Feature"] = ["A", "B", "C", "D", "E", "F"]



print("dataFrame:\n{0}\n".format(dataFrame))



print("dataFrame.loc[:, \"AGE\"]: [All rows, Begining -> \"AGE\" (inclusive) columns]\n{0}\n"

      .format(dataFrame.loc[:, "AGE"]))



print("dataFrame.loc[:, :\"SALARY\"] [All rows, Begining -> \"SALARY\" (inc) columns]\n{0}\n"

      .format(dataFrame.loc[:, :"SALARY"]))



print("dataFrame.loc[:3, \"NAME\"]: [0(inc) -> 3(inc) rows, Begining -> \"NAME\" (inc) columns]\n{0}\n"

      .format(dataFrame.loc[:3, "NAME"]))
print("dataFrame.loc[1:3, \"NAME\":\"SALARY\"]: [1(inc) -> 3(inc) rows, \"NAME\"(inc) -> \"SALARY\"(inc) columns]\n{0}\n"

      .format(dataFrame.loc[1:3, "NAME":"SALARY"]))



print("dataFrame.loc[1:2, [\"AGE\", \"SALARY\"]]: [1(inc) -> 2(inc) rows, \"AGE\" and \"SALARY\" columns]\n{0}\n"

      .format(dataFrame.loc[1:2, ["AGE", "SALARY"]]))
print("dataFrame.loc[::-1, :]: [Reversed rows, All columns]\n{0}\n"

      .format(dataFrame.loc[::-1, :]))



print("dataFrame.loc[:, 0]: [All rows, 0th column]\n{0}\n"

      .format(dataFrame.iloc[:, 0]))
#Filtering



filter_salary = dataFrame.SALARY > 2000

filtered_salary_df = dataFrame[filter_salary]

print("filtered_salary_df: [SALARY > 2000]\n{0}\n".format(filtered_salary_df))



filter_age = dataFrame.AGE > 15 

filtered_age_df = dataFrame[filter_age]

print("filtered_age_df: [AGE > 15]\n{0}\n".format(filtered_age_df))



filtered_salary_age = filter_salary & filter_age

filtered_salary_age_df = dataFrame[filtered_salary_age]

print("filtered_salary_age_df: [SALARY>2000 & AGE>15]\n{0}\n".format(filtered_salary_age_df))

#List comprehension



average_salary = dataFrame.SALARY.mean()

print("average_salary: {0} [dataFrame.SALARY.mean()]\n".format(average_salary))



dataFrame["Salary_Rate"] = [

    "Low" if each < average_salary

    else "High"

    for each in dataFrame.SALARY    

]

print("dataFrame:\n{0}\n".format(dataFrame))
print("dataFrame:\n{0}\n".format(dataFrame))



#Dropping "New Feature" column

dataFrame.drop(["New Feature"], axis=1, inplace=True)

print("dataFrame: [\"New Feature\" dropped]\n{0}\n".format(dataFrame))
print("dataFrame:\n{0}\n".format(dataFrame))



#Dropping 0th row

new_df = dataFrame.drop(0, axis=0, inplace=False)

print("new_df: [Dropped 0th row]\n{0}\n".format(new_df))
#Concatenating



data1 = dataFrame.head()

print("data1: [dataFrame.head()]\n{0}\n".format(data1))



data2 = dataFrame.tail()

print("data2: [dataFrame.tail()]\n{0}\n".format(data2))



#Vertical

data_concat_v = pd.concat([data1, data2], axis=0)

print("data_concat_v: [pd.concat([data1, data2], axis=0)]\n{0}\n".format(data_concat_v))



#Horizontal

data_concat_h = pd.concat([data1, data2], axis=1)

print("data_concat_h: [pd.concat([data1, data2], axis=1)]\n{0}\n".format(data_concat_h))
#Transforming data



dataFrame["%50-Raise"] = [(each * 1.5) for each in dataFrame.SALARY]

print("dataFrame:\n{0}\n".format(dataFrame))
#Apply method



def raise_salary(salary):

    return salary*2



dataFrame["%100-Raise"] = dataFrame.SALARY.apply(raise_salary)

print("dataFrame:\n{0}\n".format(dataFrame))