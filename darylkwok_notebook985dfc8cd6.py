from sklearn import metrics
import numpy as np

# test_pred = df['predicted_category']

# test_true= df['actual_category']


emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

y_pred = y_pred

y_act = df1

print(metrics.confusion_matrix(y_act, y_pred, labels=[0,1,2,3,4,5])) 

print(metrics.classification_report(y_act, y_pred, labels=[0,1,2,3,4,5]))
df.sample(n=3, random_state=1)

image = imread("/kaggle/working/test/0#folder/0#file name.png")

# Creating a dataset which contains just one image
image= image.reshape((1,image.shape[0], image.shape[1]))
plt.imshow(image[0], cmap=cm.gray)
!pip install scikit-plot
import scikitplot as skplt 
from sklearn.metrics import confusion_matrix
import seaborn as sns

ax = skplt.metrics.plot_confusion_matrix(y_act, y_pred, normalize=True, figsize = (10,7))

ax.set(title="Confusion Matrix",
       xticklabels =emotion_labels, yticklabels =emotion_labels,
      xlabel="Predicted",
      ylabel="Actual",)
import numpy as np
import random, os
from matplotlib.pyplot import imread, imshow, subplots, show 
import matplotlib.cm as cm

# print random image from each category
path = "/kaggle/working/test/"
w=10
h=10
fig=plt.figure(figsize=(5,5))
columns=5
rows=1
for i in range(6):
    if i!=1:
        i=str(i)
        emotion_folder=os.path.join(path,i)
        print(emotion_folder)
        random_emotion_filename=random.choice([x for x in os.listdir(emotion_folder)
                   if os.path.isfile(os.path.join(emotion_folder, x))])
        random_filepath=os.path.join(emotion_folder,random_emotion_filename)
        print(random_filepath)
        image=imread(random_filepath)
        for i in range(1,columns*rows+1):
        #print each image from each folder 
            image= image.reshape((1,image.shape[0], image.shape[1]))
            fig.add_subplot(rows,columns,i)
            plt.imshow(image[0], cmap=cm.gray)
    plt.show()