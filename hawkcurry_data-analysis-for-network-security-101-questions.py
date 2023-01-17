import pandas as pd 
df = pd.read_csv(

    '/kaggle/input/2019-trendmicro-ctf-wildcard-400/gowiththeflow_20190826.csv',

    header = 0, 

    names= ['ts', 'src', 'dst', 'port', 'bytes']

)

df.info()
answers = []
answers.append('<IP address>')
answers.append('<IP address>')
answers.append('<Port>')
answers.append('<Port>')
answers.append('<Port>')
answers.append('<IP address>')
answers.append('<IP address>')
answers.append('<Port>')
answers.append('<IP address>')
answers.append('<IP address>')
# import hashlib

# answer_hash = hashlib.md5(':'.join(answers).encode('utf-8')).hexdigest()

# assert answer_hash == 'ec766132cac80b821793fb9e7fdfd763'