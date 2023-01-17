import os

import pandas as pd

import json

import urllib



os.mkdir('./data')




birds_path = 'https://raw.githubusercontent.com/microfaune/microfaune/master/modeling/oiseaux.csv'

birds = pd.read_csv(birds_path)

count = 0



for i in range(birds.shape[0]):

    bird_query = birds.iloc[i]["Nom latin"].replace(" ", "+").lower()

    bird_name = birds.iloc[i]["Nom latin"].replace(" ", "_").lower()

    

    url = f'https://www.xeno-canto.org/api/2/recordings?query={bird_query}'

    r = urllib.request.urlopen(url)

    

    bird_data = json.loads(r.read().decode(r.info().get_param('charset') or 'utf-8'))

    

    # create folder if it doesn't exist

    dirPath = f'./data/{bird_query.replace("+", "_")}'

    print(bird_name)

    

    if not os.path.exists(dirPath):

        os.mkdir(dirPath)

    

    # retrieve recordings

    for rec in bird_data['recordings']:

        if count < 10:

            if not os.path.exists(f"./data/{bird_name}/{rec['file-name']}"):

                rec_path = f"https:{rec['file']}"

                print(rec_path)

                urllib.request.urlretrieve(rec_path, f"./data/{bird_name}/{rec['file-name']}")

            else:

                print(f"Already downloaded: ./data/{bird_name}/{rec['file-name']}")



            count += 1