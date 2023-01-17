from pandas import read_csv

data = read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()
target = "Attrition"
feature_by_dtype = {}

for c in data.columns:

    

    if c == target: continue

    

    data_type = str(data[c].dtype)

    

    if data_type not in feature_by_dtype.keys():

         feature_by_dtype[data_type] = [c]

    else:

        feature_by_dtype[data_type].append(c)



feature_by_dtype

feature_by_dtype.keys()
objects = feature_by_dtype["object"]

objects
len(objects)
for o in objects:

    text = ("%s: %s") % (o, list(data[o].unique()))

    print(text)
# Remove Features

remove = ["Over18"]
categorical_features = [f for f in objects if f not in remove]

categorical_features
int64s = feature_by_dtype["int64"]

int64s
len(int64s)
for i in [i for i in int64s if len(data[i].unique()) < 20]:

    text = ("%s: %s") % (i, list(data[i].unique()))

    print(text)
# Remove List



remove.append("StandardHours")

remove.append("EmployeeCount")

remove
# Count Features

# Features that consist of counting numbers that can also be used for binning or categorization.



count_features = []

for i in [i for i in int64s if len(data[i].unique()) < 20 and i not in remove]:

    count_features.append(i)

    

count_features
for i in [i for i in int64s if len(data[i].unique()) > 20]:

    text = ("%s: count(%s)") % (i, data[i].unique().size)

    print(text)
count_features = count_features + ["TotalWorkingYears", "YearsAtCompany", "HourlyRate"]

count_features
remove.append("EmployeeNumber")
numerical_features = [i for i in int64s if i not in remove]

numerical_features
numerical_features
data[numerical_features].head()
categorical_features
data[categorical_features].head()
count_features
data[count_features].head()
from pandas import read_csv

data = read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")



target = "Attrition"



feature_by_dtype = {}

for c in data.columns:

    

    if c == target: continue

    

    data_type = str(data[c].dtype)

    

    if data_type not in feature_by_dtype.keys():

         feature_by_dtype[data_type] = [c]

    else:

        feature_by_dtype[data_type].append(c)



objects = feature_by_dtype["object"]

remove = ["Over18"]

categorical_features = [f for f in objects if f not in remove]

int64s = feature_by_dtype["int64"]

remove.append("StandardHours")

remove.append("EmployeeCount")

count_features = []

for i in [i for i in int64s if len(data[i].unique()) < 20 and i not in remove]:

    count_features.append(i)

count_features = count_features + ["TotalWorkingYears", "YearsAtCompany", "HourlyRate"]

remove.append("EmployeeNumber")

numerical_features = [i for i in int64s if i not in remove]