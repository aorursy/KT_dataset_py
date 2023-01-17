import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as metrics



data = pd.read_csv('../input/mobile-price-classification/train.csv')

data.head()
data.columns
# Set variables for the targets and features

y = data['price_range']

X = data.drop('price_range', axis=1)



# Split the data into training and validation sets

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=7)
# Create the classifier and fit it to our training data

model = RandomForestClassifier(random_state=7, n_estimators=100)

model.fit(train_X, train_y)
# Predict classes given the validation features

pred_y = model.predict(val_X)



# Calculate the accuracy as our performance metric

accuracy = metrics.accuracy_score(val_y, pred_y)

print("Accuracy: ", accuracy)
# Calculate the confusion matrix itself

confusion = metrics.confusion_matrix(val_y, pred_y)

print(f"Confusion matrix:\n{confusion}")





# Normalizing by the true label counts to get rates

print(f"\nNormalized confusion matrix:")

for row in confusion:

    print(row / row.sum())
probs = model.predict_proba(val_X)

print(probs)