# this is a clone of https://www.kaggle.com/miklgr500/how-to-use-translators-for-comments-translation but for google colab
from google.colab import files



# Install Kaggle library

!pip install -q kaggle
import multiprocessing

print(multiprocessing.cpu_count())
# Upload kaggle API key file

uploaded = files.upload()
!ls
!mkdir ~/.kaggle

!cp /content/kaggle.json ~/.kaggle/kaggle.json
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()

api.authenticate()

api.competition_download_file('jigsaw-multilingual-toxic-comment-classification','jigsaw-unintended-bias-train.csv')
!ls
!unzip jigsaw-unintended-bias-train.csv.zip
! ls
!pip install translators



import pandas as pd

# current version have logs, which is not very comfortable

import translators as ts

from multiprocessing import Pool

from tqdm import *
LANG = 'ru'

API = 'google'





def translator_constructor(api):

    if api == 'google':

        return ts.google

    elif api == 'bing':

        return ts.bing

    elif api == 'baidu':

        return ts.baidu

    elif api == 'sogou':

        return ts.sogou

    elif api == 'youdao':

        return ts.youdao

    elif api == 'tencent':

        return ts.tencent

    elif api == 'alibaba':

        return ts.alibaba

    else:

        raise NotImplementedError(f'{api} translator is not realised!')

CSV_PATH = 'jigsaw-unintended-bias-train.csv'



def translate(x):

    try:

        return [x[0], translator_constructor(API)(x[1], 'en', LANG), x[2]]

    except:

        return [x[0], None, [2]]





def imap_unordered_bar(func, args, n_processes: int = 48):

    p = Pool(n_processes, maxtasksperchild=100)

    res_list = []

    with tqdm(total=len(args)) as pbar:

        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):

            pbar.update()

            res_list.append(res)

    pbar.close()

    p.close()

    p.join()

    return res_list





def main():

    df = pd.read_csv(CSV_PATH).query('toxic==1') # .sample(100)

    df.toxic = df.toxic.round().astype(int)

    tqdm.pandas('Translation progress')

    df[['id', 'comment_text', 'toxic']] = imap_unordered_bar(translate, df[['id', 'comment_text', 'toxic']].values)

    df.to_csv(f'jigsaw-toxic-comment-train-{API}-{LANG}.csv')





if __name__ == '__main__':

    import multiprocessing

    print(multiprocessing.cpu_count())

    main()