import pandas
import numpy

data = pandas.read_csv("../input/train_V2.csv")
# sorted_data = data.sort_values("matchId")
data = data.dropna()
data = data.drop(["Id", "groupId", "matchId", "matchType"], axis="columns")
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(data.values, test_size=0.2)
train = data.values

test = pandas.read_csv("../input/test_V2.csv")
test_id = test.Id
test = test.drop(["Id", "groupId", "matchId", "matchType"], axis="columns")
test = test.values
test = numpy.column_stack((test, numpy.zeros(test.shape[0])))
from sklearn.ensemble import GradientBoostingRegressor

X = train[:, :-1]
Y = train[:, -1]
predictor = GradientBoostingRegressor(n_estimators=1000)
predictor.fit(X, Y)
x = test[:, :-1]
y = test[:, -1]
m = x[:, 12]-1

z = predictor.predict(x)
z = numpy.around((1.0/m)*(z*m).astype(numpy.int16), decimals=4)
# print(numpy.sum(numpy.abs(z-y)))
# print(y[:10])
# print(z[:10])
with open("submission.csv", "w") as file:
    file.write("Id,winPlacePerc\n")
    for i in range(len(test_id)):
        file.write(test_id[i]+","+str(z[i])+"\n")