from PIL import Image
Image.open('../input/landmark-recognition-2020/test/0/0/0/000b15b043eb8cf0.jpg')
Image.open('../input/landmark-recognition-2020/test/0/0/0/00084cdf8f600d00.jpg')
!wget https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv
TRAIN_LABEL_TO_CATEGORY = 'train_label_to_category.csv'



GAIA_CATEGORIES = [

    'http://commons.wikimedia.org/wiki/Category:I_complessi',

    'http://commons.wikimedia.org/wiki/Category:Films_by_Luigi_Filippo_D%27Amico',

    'http://commons.wikimedia.org/wiki/Category:1965_films',

    'http://commons.wikimedia.org/wiki/Category:Gaia_Germani'

]



TOWER_CATEGORIES = [

    'http://commons.wikimedia.org/wiki/Category:Tour_Saint-Nicolas'

]
import pandas as pd
df = pd.read_csv(TRAIN_LABEL_TO_CATEGORY)
landmark_categories = set(df.category.values)
gaia_is_landmark = False



for cat_name in GAIA_CATEGORIES:

    if cat_name in landmark_categories:

        gaia_is_landmark = True



if gaia_is_landmark:

    print('Gaia Germani IS a landmark')

else:

    print('Gaia Germani IS NOT a landmark')
tower_is_landmark = False



for cat_name in TOWER_CATEGORIES:

    if cat_name in landmark_categories:

        tower_is_landmark = True



if tower_is_landmark:

    print('Saint Nicolas Tower IS a landmark')

else:

    print('Saint Nicolas Tower IS NOT a landmark')