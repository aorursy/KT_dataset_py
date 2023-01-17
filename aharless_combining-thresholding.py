BASE_THRESH = .2

BASE_WEIGHT = .40
import numpy as np  

import pandas as pd

from tqdm import tqdm
probs9187 = pd.read_csv("amaGiannakProb4e4d7e5cv9187.csv")

probs9203 = pd.read_csv("amaGiannakProb5e4d1e4cv9203.csv")

probs9193 = pd.read_csv("amaGiannakProb5e4d8e5cv9193.csv")

probs9222 = pd.read_csv("vgg19prob00025d0008cv9222.csv")

probs9220 = pd.read_csv("vgg19prob0001d001cv9220.csv")

probs9259 = pd.read_csv("vgg19prob00007d005cv9259.csv")

probs9263 = pd.read_csv("vgg19prob00007d0023cv9263.csv")

probs9269 = pd.read_csv("vgg19prob00006d0028cv9269.csv")

probs9281 = pd.read_csv("vgg19prob00005d0028cv9281.csv")

probs9285 = pd.read_csv("vgg19prob128r000044d0028cv9285.csv")

probs9268 = pd.read_csv("vgg19prob144r000044d0028cv9268.csv")

probs9263a = pd.read_csv("vgg19prob128r00004d0028cv9263.csv")

probs9256new = pd.read_csv("vgg19prob128r00005d003cv9256new.csv")

probs9238newtta = pd.read_csv("vgg19probTta136r000046d003cv9238new.csv")

probs9245newtta = pd.read_csv("vgg19probTta128r000047d0029cv9245new.csv")

df_test_data = pd.read_csv(r"d:\bigdata\amazon\sample_submission_v2.csv")
optThresh={ # Average mine and Heng CherKeng

'clear': 0.18,

'water': 0.19,

'artisinal_mine': 0.20,

'cultivation': 0.20,

'selective_logging': 0.14,

'bare_ground': 0.16,

'habitation': 0.18,

'partly_cloudy': 0.13,

'conventional_mine': 0.21,

'haze': 0.15,

'road': 0.17,

'cloudy': 0.06,

'agriculture': 0.18,

'slash_burn': 0.12,

'primary': 0.22,

'blooming': 0.20,

'blow_down': 0.06

}
preds = []



for i in tqdm(range(probs9285.shape[0]), miniters=1000):

    a = ( .0100 * probs9193.iloc[[i]] +

          .0100 * probs9203.iloc[[i]] + 

          .0200 * probs9220.iloc[[i]] +

          .0200 * probs9222.iloc[[i]] +

          .1600 * probs9238newtta.iloc[[i]] + 

          .1800 * probs9245newtta.iloc[[i]] + 

          .0800 * probs9256new.iloc[[i]] +

          .0500 * probs9259.iloc[[i]] +

          .0500 * probs9263.iloc[[i]] +

          .0600 * probs9263a.iloc[[i]] +

          .0600 * probs9268.iloc[[i]] +

          .0800 * probs9269.iloc[[i]] +

          .1050 * probs9281.iloc[[i]] +

          .1150 * probs9285.iloc[[i]] )

    labels = ""

    for i in range(a.shape[1]):

        p = a.iloc[0,i]

        n = a.columns[i]

        thresh = BASE_WEIGHT*BASE_THRESH + (1-BASE_WEIGHT)*optThresh[n]

        if( p > thresh ):

            if labels=="":

                labels = n

            else:

                labels += " " + n

    preds.append(labels)

df_test_data['tags'] = preds

df_test_data.to_csv('comboOptThresh40c.csv', index=False)