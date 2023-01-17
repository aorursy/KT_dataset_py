import pandas as pd
data = pd.DataFrame({'apple': [

    'りんご/ごりら/らっぱ/ぱんだ',

    'りんご/ごりら/らっこ',

    'りんご/ごりら/らっぱ'

]})
data
data['cnt'] = data['apple'].str.count('/') + 1
data