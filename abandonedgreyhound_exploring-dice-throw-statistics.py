# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Constants
dice_sides_count = 6
number_of_dice = 2
number_of_throws = 10000
rolls = pd.DataFrame(np.random.randint(low = 1, high = dice_sides_count + 1,
                                       size = (number_of_throws, number_of_dice)))
rolls.columns = ["Die " + str(i + 1) for i in range(number_of_dice)]
rolls = rolls.assign(sum = rolls.sum(axis = 1))
rolls.describe()
sns.countplot(x = rolls['sum'])

