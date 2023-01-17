import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr
data = pd.read_csv('../input/train.csv')
pr(data.Fare, data.Pclass)
plt.style.use('bmh')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Age vs Survival')
plt.hist(data.Age[(np.isnan(data.Age) == False)], bins= 15, alpha = 0.4, color = 'r', label = 'Before')
plt.hist(data.Age[(np.isnan(data.Age) == False) & (data.Survived == 1)], bins= 15, alpha = 0.4, color = 'b', label = 'After')
#plt.hist(data.Age[data.Age != np.NaN])
plt.legend(loc = 'upper right')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr
data = pd.read_csv('../input/train.csv')
pr(data.Fare, data.Pclass)
plt.style.use('bmh')
plt.xlabel('No. of Sibling/Spouse')
plt.ylabel('Survived')
plt.title('SibSp vs Survival')
plt.hist(data.SibSp, label = 'before', alpha = 0.4, color = 'b')
plt.hist(data.SibSp[data['Survived'] == 1], label = 'after', alpha = 0.4, color = 'r')
plt.legend(loc = 'upper right')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr
data = pd.read_csv('../input/train.csv')
pr(data.Fare, data.Pclass)
plt.style.use('bmh')
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.title('Sex vs Survived')
plt.hist(data.Sex, color = 'b', alpha = 0.4, label = 'before')
plt.hist(data[data['Survived'] == 1].Sex, color = 'r', alpha = 0.4, label = 'after')
plt.legend(loc = 'upper center')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr
data = pd.read_csv('../input/train.csv')
pr(data.Fare, data.Pclass)
plt.style.use('bmh')
plt.xlabel('Class')
plt.ylabel('Fare')
plt.title('Class vs Fare')
data_to_plot = [data[data['Pclass'] == 1].Fare.values, data[data['Pclass'] == 2].Fare.values, data[data['Pclass'] == 3].Fare.values]
plt.boxplot(data_to_plot)
plt.xticks([1,2,3], ['First', 'Second', 'Third'])
plt.show()
#Exploration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv('../input/train.csv')
plt.style.use('bmh')
plt.xlabel("PClass")
plt.xticks([1,2,3], ['First', 'Second', 'Third'])
plt.title("Survival vs PClass")
plt.hist(data.Pclass, color = 'b', alpha = 0.4, label = 'before')
plt.hist(data.Pclass[data.Survived == 1], color = 'r', alpha = 0.4, label = 'after')
plt.legend(loc = 'upper left')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
data = pd.read_csv('../input/train.csv')
plt.style.use('bmh')

data.describe()
#data.SibSp.value_counts()

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import os

#Preparing Training Data
data = pd.read_csv('../input/train.csv')
data.Age.fillna(value = np.round(np.mean(data.Age)), inplace = True)
input_data = pd.get_dummies(data.drop(['Cabin', 'Survived', 'Name', 'Ticket', 'PassengerId'], axis = 1))
target_data = data['Survived']
#Preparing Test Data
test_data = pd.read_csv('../input/test.csv')
test_data['Age'].fillna(value = np.mean(test_data['Age']), inplace = True)
test_data['Fare'].fillna(value = np.mean(test_data['Fare']), inplace = True)
input_test_data = pd.get_dummies(test_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis = 1))
#model
param_grid = {'max_features': [0.3, 0.5, 0.7, 1.0]}
dt = DecisionTreeClassifier()
model = GridSearchCV(dt, param_grid, cv = 5)
model.fit(input_data, target_data)
survived = model.predict(input_test_data)
p_id = test_data['PassengerId']
result_data = pd.DataFrame({'Survived': survived}, index = p_id)
result_data.to_csv('output_dt.csv')
print("Decision Tree Accuracy (5-fold CV): ", model.best_score_)
