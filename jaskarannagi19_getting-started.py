import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print('Read data...')
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1].values.ravel()
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
print('Training ...')
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_images, train_labels)


print('Predicting...')
test_data = pd.read_csv('../input/test.csv')
prd = rf.predict(test_data)

print (prd)
print('Saving...')
output = pd.DataFrame({"ImageId": list(range(1, len(prd) + 1)), "Label": prd})
output.to_csv("submission.csv", index=False, header=True)



