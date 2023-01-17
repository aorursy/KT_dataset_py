import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())

# Your code here
import pandas as pd
pd.DataFrame([{'Apples':30,"Bananas":21}])
#print(answer_q1())
# Your code here
import pandas as pd
pd.DataFrame([{'Apples':30,"Bananas":21}],index=["2017  Sales", "2018 Sales"])
# Your code here
import pandas as pd
pd.Series({'Flour':'4 cups','Milk':'1 cup', 'Eggs':'2 large','Spam':'1 can'}, name="Dinner")
#print(answer_q3())
# Your code here 
wineMag = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
wineMag
# Your code here 
import pandas as pd
wineMag = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name="Pregnant Women Participating")
wineMag
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
q6_df.to_csv('cows_and_goats.csv') 
# Your Code Here