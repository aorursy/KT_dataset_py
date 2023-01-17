import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pycaret.classification import *
dataset = pd.read_pickle('/kaggle/input/esg-factor-returns/esg_factor_returns.pkl')
dataset.index = range(0,len(dataset))
data = dataset.sample(frac=0.99, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
#for training purposes let's assume we know the future
data['SDG_tomorrow'] = data.SDG.shift(-1)
data = data.dropna()
#data.tail()
# let's label a day with positive performance "good", and negativ "bad"
data.loc[data['SDG_tomorrow'] >=0, 'target'] = "long tomorrow"
data.loc[data['SDG_tomorrow'] < 0, 'target'] = "short tomorrow"
data = data.drop(columns=['SDG_tomorrow'])
data.tail()
clf = setup(data = data, target = 'target', session_id=123) 
compare_models()
lda = create_model('lda')
tuned_lda = tune_model(lda)
evaluate_model(tuned_lda)
plot_model(tuned_lda, plot = 'auc', save=True)
plot_model(tuned_lda, plot = 'confusion_matrix', save=True)
plot_model(tuned_lda, plot = 'feature', save=True)
predict_model(tuned_lda);
final_lda = finalize_model(tuned_lda)
print(final_lda)
predict_model(final_lda);
unseen_predictions = predict_model(final_lda, data=data_unseen)
unseen_predictions.head()
save_model(final_lda,'Final LDA Model 09Aug2020')
loaded_model = load_model('Final LDA Model 09Aug2020')
new_prediction = predict_model(loaded_model, data=data_unseen)
new_prediction
