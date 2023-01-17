# import pandas 

import pandas as pd 
pd.read_csv("../input/archive.csv")
data = pd.read_csv("../input/archive.csv")

data.describe()



data.describe().transpose()




# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.