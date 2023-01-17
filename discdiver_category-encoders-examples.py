# A notebook for exploring the contrasts created by each category_encoders option
## By Jeff Hale
import numpy as np
import pandas as pd              # version 0.23.4
import category_encoders as ce   # version 1.2.8
from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = '{:.2f}'.format # to make legible

# make some data
df = pd.DataFrame({
    'color':["a", "c", "a", "a", "b", "b"], 
    'outcome':[1, 2, 0, 0, 0, 1]})

# set up X and y
X = df.drop('outcome', axis = 1)
y = df.drop('color', axis = 1)
print(X) 

le = LabelEncoder()
encoded = le.fit_transform(np.ravel(X))    # warning thrown without np.ravel

print("\n The result of transforming X with LabelEncoder:")
print(encoded)
print(type(encoded))
X
ce_ord = ce.OrdinalEncoder(cols = ['color'])
ce_ord.fit_transform(X, y['outcome'])
ce_one_hot = ce.OneHotEncoder(cols = ['color'])
ce_one_hot.fit_transform(X, y)
ce_bin = ce.BinaryEncoder(cols = ['color'])
ce_bin.fit_transform(X, y)
ce_basen = ce.BaseNEncoder(cols = ['color'])
ce_basen.fit_transform(X, y)
ce_hash = ce.HashingEncoder(cols = ['color'])
ce_hash.fit_transform(X, y)
X
import numpy as np
import pandas as pd              # version 0.23.4
import category_encoders as ce   # version 1.2.8

pd.options.display.float_format = '{:.2f}'.format # to make legible

# some new data frame for the contrast encoders
df2 = pd.DataFrame({
    'color':["a", "b", "c", "d"], 
    'outcome':[1, 2,  0, 1]})

# set up X and y
X2 = df2.drop('outcome', axis = 1)
y2 = df2.drop('color', axis = 1)
X2
ce_helmert = ce.HelmertEncoder(cols = ['color'])
ce_helmert.fit_transform(X2, y2)
ce_sum = ce.SumEncoder(cols = ['color'])
ce_sum.fit_transform(X2, y2)
ce_backward = ce.BackwardDifferenceEncoder(cols = ['color'])
ce_backward.fit_transform(X2, y2)
ce_poly = ce.PolynomialEncoder(cols = ['color'])
ce_poly.fit_transform(X2, y2)
import numpy as np
import pandas as pd              # version 0.23.4
import category_encoders as ce   # version 1.2.8

pd.options.display.float_format = '{:.2f}'.format # to make legible

# some new data frame for the contrast encoders
df3 = pd.DataFrame(
    {'color':[3,2,0,1,1,1,2,2,3,3,4,5,6,7,8,3], 
     'outcome':[1,0,0,1,2,1,1,0,0,1,0,2,2,1,1,1]})

# set up X and y
X3 = df3.drop('outcome', axis = 1)
y3 = df3.drop('color', axis = 1)

df3
# Target with default parameters
ce_target = ce.TargetEncoder(cols = ['color'])

ce_target.fit(X3, y3['outcome'])
# Must pass the series for y in v1.2.8

ce_target.transform(X3, y3['outcome'])
# Target with min_samples_leaf higher
ce_target_leaf = ce.TargetEncoder(cols = ['color'], min_samples_leaf = 10)

ce_target_leaf.fit(X3, y3['outcome'])
# Must pass the series for y in v1.2.

ce_target_leaf.transform(X3, y3['outcome'])
# Target with smoothing higher
ce_target_leaf = ce.TargetEncoder(cols = ['color'], smoothing = 10)
ce_target_leaf.fit(X3, y3['outcome'])
# Must pass the series for y in v1.2.b
ce_target_leaf.transform(X3, y3['outcome'])
# Target with smoothing higher
ce_target_leaf = ce.TargetEncoder(cols = ['color'], smoothing = .1)
ce_target_leaf.fit(X3, y3['outcome'])
# Must pass the series for y in v1.2.b
ce_target_leaf.transform(X3, y3['outcome'])
ce_leave = ce.LeaveOneOutEncoder(cols = ['color'])
ce_leave.fit(X3, y3['outcome'])        
ce_leave.transform(X3, y3['outcome'])         
# Must pass the series for y
# Fit and transform need to be called seperately for LeaveOneOut until v1.3.0
# ce_leave = ce.WeightOfEvidenceEncoder(cols = ['color'])
# ce_leave.fit(X3, y3['outcome'])        
# ce_leave.transform(X3, y3['outcome'])         
# Must pass the series for y
# Fit and transform need to be called seperately for LeaveOneOut until v1.3.0