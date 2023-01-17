import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_data = pd.read_csv("/kaggle/input/figure-out-the-formula-dogfooding/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/figure-out-the-formula-dogfooding/test.csv")

test_data.head()
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=3)



train_data = train_data.drop("Id", axis=1)

x = train_data.drop("Expected", axis=1)

y = train_data["Expected"]

knn.fit(x, y)



print("Done training")
solutions = []



for index, row in test_data.iterrows():

    row_id = row["Id"]

    variables = row["a":"h"]

    prediction = knn.predict([variables])

    print("Prediction is " + str(prediction))

    solutions.append([row_id, prediction[0]])

    

solutions
submission = pd.DataFrame(solutions, columns=["Id", "Expected"])

submission.to_csv("submission.csv", index=False)