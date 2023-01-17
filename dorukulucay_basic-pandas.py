import pandas as pd


fellowship = {
  "NAME": ["Gandalf", "Aragorn", "Legolas", "Gimli", "Boromir", "Frodo", "Sam", "Merry", "Pippin"],
  "AGE": [10000, 87, 500, 140, 40, 50, 35, 35, 35],
  "HITPOINT": [1000, 100, 50, 150, 110, 30, 30, 30, 30],
  "SPEED": [1000, 90, 200, 90, 80, 100, 100, 100, 100],
  "ATTACK": [1000, 120, 80, 150, 130, 15, 15, 15, 15],
  "CLASS": ["WIZARD", "MAN", "ELF", "DWARF", "MAN", "HOBBIT", "HOBBIT", "HOBBIT", "HOBBIT"]
}

print(type(fellowship))

dt = pd.DataFrame(fellowship)

print(type(dt))
dt.head()
dt.head(3)
dt.tail(3)
dt.info()
dt.describe()
dt.corr()
dt2 = dt[0:5]
dt2
dt["CLASS"]
# we create a filter to give us only the rows whichs AGE field is greater than 100
filter_age = dt.AGE > 100

# if we want to see what is in filter, we will see a serie(vector) that contains the filters result
filter_age
# here we apply the filter to our dataframe and see the filtered data
filteredData = dt[filter_age]

filteredData
dt["OLDER_THAN_100"] = ["True" if each > 100 else "False" for each in dt.AGE]

dt
# an alternative using the filter we already created before
dt["OLDER_THAN_100_ALTERNATIVE"] = filter_age 

dt
# because we have two cols that do the same thing, we remove one
dt.drop(["OLDER_THAN_100_ALTERNATIVE"], axis=1, inplace=True) #i'll explain inplace later

dt
a = 5
# to add 4 to a
a = a + 4
dtf = pd.DataFrame({"TEST":[1,2,3,4], "TOBEDROPPED": [3,4,5,6], "TOBEDROPPED2" : [5,6,7,8]})
dtf
# now, if we want to drop the col TOBEDROPPED, we'd usually do
dtf = dtf.drop(['TOBEDROPPED'], axis=1)
dtf
# but pandas has a better solution to it. 
# When you pass inplace option true, it does the change directly to your dataframe
dtf.drop(['TOBEDROPPED2'], axis=1, inplace=True)
dtf
print("average age of fellowship is "+ str( dt.AGE.mean()))
# concat vertical
dtFirstAndLast = pd.concat([dt.tail(1), dt.head(1)], axis=0)
dtFirstAndLast
#concat horizontal
dtNames = pd.concat([dt.NAME], axis =1)
dtNames
def is_of_man_race(_class):
    return _class == "MAN"

# transformation via apply
dt["IS_MAN"] = dt.CLASS.apply(is_of_man_race)
dt