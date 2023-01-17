import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
print(reviews)
# Your code here
# 将原来的数据列名为region_1和region_2分别替换为region和locale
reviews_rename_col=reviews.rename(columns={'region_1':'region','region_2':'locale'})
check_q1(reviews_rename_col)
reviews_rename_col.head()
# Your code here
# 将行号名称设置为wines
reviews_rename_rows = reviews.rename_axis('wines',axis='rows')
check_q2(reviews_rename_rows)

reviews_rename_rows.head()
reviews_rename_axis = reviews.rename_axis('wines',axis='rows').rename_axis('fields',axis='columns')
reviews_rename_axis.head()
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
# 合并两个workbook
gaming_movie = pd.concat([gaming_products,movie_products])
check_q3(gaming_movie)
gaming_movie.head()
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
# 查看原始数据powerlifting_meets
powerlifting_meets
# 查看原始数据powerlifting_competitors
powerlifting_competitors
# Your code here
#powerlifting_join = powerlifting_meets.set_index('MeetID').join(powerlifting_competitors.set_index('MeetID'))
powerlifting_join = powerlifting_meets.join(powerlifting_competitors.set_index('MeetID'))
ans = powerlifting_join.set_index('MeetID')
ans
# 可整合上述代码至一行
powerlifting_join2 = powerlifting_meets.join(powerlifting_competitors.set_index('MeetID')).set_index('MeetID')
powerlifting_join2
# 对结果进行检验
check_q4(ans)
# 对第二种方法的答案进行检验
check_q4(powerlifting_join2)