import numpy as np
import pandas as pd
DATA_DIR = '../input/'
train = pd.read_csv(DATA_DIR+'train.csv')
train.index=train.ID

original_columns = ['target','f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
                   '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
                   'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
                   '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212',  '66ace2992',
                   'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
                   '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
                   '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2',  '0572565c2',
                   '190db8488',  'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'] 

new_columns = ['f3cf9341c','fa11da6df','d47c58fe2','555f18bd3','134ac90df','716e7d74d','1bf8c2597',
               '1f6b2bafa','174edf08a','5bc7ab64f','a61aa00b0','b2e82c050','26417dec4',
               '51707c671','e8d9394a0','cbbc9c431','6b119d8ce','f296082ec','be2e15279',
               '698d05d29','38e6f8d32','93ca30057','7af000ac2','1fd0a1f2a','41bc25fef',
               '0df1d7b9a','2b2b5187e','bf59c51c3','cfe749e26','ad207f7bb','11114a47a',
               'a8dd5cea5','b88e5de84']

rows = ['19bafc0eb','8ef819e43','5342778cc','96b1c70cf','359d25b38','c71842c11',
        'ccfb14bf1','9d9176f62','c989a1ff0','7d4b5b291','102bd04d8','9eb312e66',
        '557b8a17e','eb13cb6a1','fa5455b9c','1a2fcf1b6','1a4fdc864','1a060c889',
        'c114630ea','3bf4d0dd2','87daaf67b','7e8d06e00','0dadd8851','e8b380c2a',
        '12b150758','9db0f2cd6','802f25a23','59f52b75a','1e49872f3','4fa3167f8',
        'bb5041e89','f2a11f83b','f84109ff1','47c4e8ec2','6b889a750','3540b1ba4',
        '71286b738','de5f122d8','fa93b66c1','e1fffe216','693802083','8e8c910ce',
        '3bd8c8e64','319e95726','5ea3ca394','720907682','d08699be7','97ebb131c',
        '499520a32','d49a3bd3e','571d2ea50','a74e200d2','bf99827e7','e6139f9c2',
        '2e1ce6102','8eb4c28c7','96630c88a','15defc71c','d0a52a9a5','917a29ba4',
        'eec39f669','4eb3978dd','0110c05db','f11ffbbaf','36153f507','86f59f6c9',
        '18516f2bb','a1f713030','646e59743','e677c32f6','52bdaa610','17224efee',
        '2d6fff5cc','7ec5a9a38','376d7dced','22cd3d13a','688b43e3e','f637c0f24',
        '810285ac0','1653b322d','ef3f27c0c','ef111073a','3499dd5b0','987541b9c',
        '6cf4b79b1','567a57827','3f5a7464e','d339b8d29','0f2e84a29','b4e4afdf5',
        '815281bbe','96e9102e5','e375d6c21','08d27660b','67c967a26','3fb98b688',
        'b82c7eef3','a37b27d20','b688e11af','cb7691896','7ba151fb4']
train.loc[rows, new_columns]
train.loc[rows, original_columns]