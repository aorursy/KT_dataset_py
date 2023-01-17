import matplotlib.pyplot as plt
# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/add_custom_package.jpg.py')
fig = plt.figure(figsize = (70,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)
# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/add_repo_details.jpg.py')
plt.figure(figsize = (10,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)
# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/installed_package.jpg.py')
plt.figure(figsize = (10,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)
# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/console.jpg.py')
plt.figure(figsize = (30,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)
# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/uninstall_package.jpg.py')
plt.figure(figsize = (20,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)
from keras.models import load_model
# Read model from the model file
model = load_model('/opt/conda/lib/python3.6/site-packages/mytest/covert_keras_model.py')
# Verify that it works
model.summary()  # Print model summary
import matplotlib.pyplot as plt
#Read image from disk
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/covert_legend_kid_jpg.py')

# Display image
plt.figure(figsize = (30,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img, interpolation='nearest')
import pandas as pd
df = pd.read_csv('/opt/conda/lib/python3.6/site-packages/mytest/covert_leaf_classification_csv.py')
df.head()
