import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
data = np.load('../input/solan-digital-sky-sarvey-galaxycolors/sdss_galaxy_colors (3).npy')

data
def get_features_target(data):
    feature = np.zeros(shape=(len(data), 4))
    feature[:, 0] = data['u'] - data['g']
    feature[:, 1] = data['g'] - data['r']
    feature[:, 2] = data['r'] - data['i']
    feature[:, 3] = data['i'] - data['z']
    target = data['redshift']
    return feature, target
feature , target = get_features_target(data)
target
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.1)
#y_train.size
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
predictions = dtr.predict(x_test)
predictions
y_test
dtr.score(feature,target)
from sklearn.metrics import precision_score

    
cmap = plt.get_cmap('YlOrRd')

    # colour indexes u-g and r-i
u_g = data['u'] - data['g']
r_i = data['r'] - data['i']
    # redshift array
redshift = data['redshift']
    # plot with plt.scatter and plt.colorbar
plot = plt.scatter(u_g, r_i, s=0.8, lw=0, c=redshift, cmap=cmap)

cb = plt.colorbar(plot)
cb.set_label('Redshift')
    
plt.xlabel('Colour index  u-g')
plt.ylabel('Colour index  r-i')
plt.title('Redshift (colour) u-g versus r-i')
    #axis limits
plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 1)
plt.show()
def median_diff(x, y):
    ans = np.median(np.abs(x-y))
    return ans
def accuracy_by_treedepth(feature, target, depths):
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.1)
    train = []
    test = []
    for depth in depths:
        dtr = DecisionTreeRegressor(max_depth=depth)
        dtr.fit(x_train, y_train)
        prediction = dtr.predict(x_train)
        train.append(median_diff(y_train, prediction))
        
        prediction = dtr.predict(x_test)
        test.append(median_diff(y_test, prediction))
    return train, test
        
tree_depth = [i for i in range(1,36,2)]
train_mean, test_mean = accuracy_by_treedepth(feature, target, tree_depth)
print(train_mean)
train_plot = plt.plot(tree_depth, train_mean, label='Training set')
test_plot = plt.plot(tree_depth, test_mean, label='Testing set')
plt.xlabel("maximum tree depth")
plt.ylabel("median difference")
plt.legend()
plt.show()
