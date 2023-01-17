import pandas as pd
from sklearn import linear_model

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#print(train.describe())

train_data = train.loc[:100, ["Store", "DayOfWeek", "Customers", "Open", "Promo", "SchoolHoliday"]].values
print(train_data.shape)
train_label = train.loc[:100, "Sales"].values
clf = linear_model.Lasso(alpha=.5)
clf.fit(train_data, train_label)

plot(train_label, clf.predict(train_data))

