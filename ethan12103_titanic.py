import os
import math
import numpy as np 
import pandas as pd 

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
DATA_FOLDER = "Titanic"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, DATA_FOLDER, "images")

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
datapath = os.path.join(PROJECT_ROOT_DIR, DATA_FOLDER)
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv", thousands=',')
print(titanic_train)
survived = titanic_train.loc[titanic_train['Survived']==1, :]['Sex'].value_counts()
died = titanic_train.loc[titanic_train['Survived']==0, :]['Sex'].value_counts()
train_plt = pd.DataFrame([survived,died])
train_plt.index=['Survived','Died']
train_plt.plot(kind='bar', stacked=True, title='Survival Rate Between Sexes')
titanic_train["Sex"]= titanic_train["Sex"].replace("female", 1)
titanic_train["Sex"]= titanic_train["Sex"].replace("male", 0)
print(titanic_train)
b=titanic_train.drop("Name", axis=1)
z=b.drop("Ticket", axis=1)
e=z.drop("Cabin", axis=1)
d=e.drop("Embarked", axis=1)
x=d.drop("Age", axis=1)
y = titanic_train["Survived"]
plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 100, 0, 2])
plt.show()
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
lin_reg.intercept_, lin_reg.coef_
data = x.iloc[:5]
labels = y.iloc[:5]
from sklearn.metrics import mean_squared_error
predictions = lin_reg.predict(x)
lin_rmse = np.sqrt(mean_squared_error(y,predictions))
lin_rsme