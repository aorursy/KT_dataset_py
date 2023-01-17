#import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

%matplotlib inline

#useful function to write results
def write_results(predicted, name = 'predicted.csv'):
    df = pd.DataFrame(predicted)
    df.index.name='ImageId'
    df.index+=1
    df.columns=['Label']
    df['Label']=df['Label'].astype(int)
    df.to_csv(name, header=True)
    
# useful function to plot digits
def plot_digits(list):
    plt.figure(figsize=(20,20))
    nn=0

    for idx in list:
        nn+=1
        plt.subplot(10,10,nn)
        plt.title("label:{}".format(y_df.iloc[idx]))
        #plt.title("index: {}".format(img.index[count]))
        grid_data = X_df.iloc[idx].values.reshape(28,28)
        plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")
        plt.xticks([])
        plt.yticks([])
    plt.show()
#########
labeled_images = pd.read_csv('../input/train.csv')
X_df = labeled_images.iloc[0:,1:]
y_df = labeled_images.iloc[0:,0]
#split to train and test data
train_images, test_images,train_labels, test_labels = train_test_split(X_df, y_df, train_size=0.67, random_state=1)
pipe0 = Pipeline([("pca",PCA(whiten=True)),("clf", svm.SVC(kernel='rbf'))])
### grid of parameters I tested. The best results are obtained with PCA n_components=45, C=5, gamma=0.02. 
#params={'pca__n_components': [25, 30, 35, 40, 45, 50],'clf__C': [1, 2, 5, 10, 30, 60], 'clf__gamma': [0.02]}
###
params={'pca__n_components': [45],'clf__C': [1], 'clf__gamma': [0.02]}
grid_search = GridSearchCV(pipe0, param_grid = params, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_images,train_labels.values.ravel())

prediction0_test=grid_search.predict(test_images)
prediction0_train=grid_search.predict(train_images)

print('Grid best parameter (max. accuracy): ', grid_search.best_params_)
print('Grid best score (accuracy): ', grid_search.best_score_)
print('Test score (accuracy): ', grid_search.score(test_images, test_labels.values.ravel()))
mask_wrong_train=(prediction0_train!=train_labels.values.ravel())
print('the model fails on {} digits out of {}'.format(mask_wrong_train.sum(), len(train_labels)))
list=[]
for num in np.arange(0,10):
    train_labels_wrong=train_labels[mask_wrong_train]
    flag=(train_labels_wrong==num)
    for i in train_labels_wrong.index[flag][:10]:
        list.append(i)

plot_digits(list)
plot_digits([7505, 7362, 6515, 24499, 18534, 23604, 23237, 33466, 26479, 31649, 7514, 28560, 7810, 19634])
plot_digits([16124, 38159, 29538, 21695, ])
plot_digits([23299, 41691, 25172, 15219, 5747, 4226, 25946])
pipe = Pipeline([("pca",PCA(n_components=45, whiten=True)),("clf", svm.SVC(kernel='rbf', C=5, gamma=0.02))])

# fit the model
pipe.fit(X_df,y_df.values.ravel())
# upload the test set
Xtest=pd.read_csv('../input/test.csv')
predicted_test=pipe.predict(Xtest)
write_results(predicted_test, name='predicted.csv')
