import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





features = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']





train_data = pd.read_csv("/kaggle/input/figure-out-the-formula-dogfooding/train.csv")

train_data.head()



train_x = train_data[features]

train_y = train_data.Expected
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=1)



dt.fit(train_x, train_y)



print("Done training")
test_data = pd.read_csv("/kaggle/input/figure-out-the-formula-dogfooding/test.csv")

test_data.head()



test_x = test_data[features]



print("Making predictions for the following 5 inputs:")

print(test_x.head())

print("The predictions are")

print(dt.predict(test_x.head()))
predictions = dt.predict(test_x)

    

# Associate each item with an ID (that happens to be an index)

solutions = [[i,v] for i,v in enumerate(predictions)]
submission = pd.DataFrame(solutions, columns=["Id", "Expected"])

submission.to_csv("submission.csv", index=False)



print("all done")