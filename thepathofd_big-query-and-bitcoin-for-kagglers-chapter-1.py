import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import bigquery
from bq_helper import BigQueryHelper
import squarify
import time
import gc
import re
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
bq_assistant = BigQueryHelper('bigquery-public-data','bitcoin_blockchain',max_wait_seconds=6000)
bq_assistant.list_tables()
n=5000
A=bq_assistant.head("transactions", num_rows=n)
A.head()
pd.DataFrame(A.inputs.iloc[0])
pd.DataFrame(A.outputs.iloc[0])
print('----------- redeem script (input) -------------')
print('PUSHDATA(71)[sig]')
print('----------- locking script (output) -----------')
print('PUSHDATA(33)[pubkey]\nCHECKSIG')
script=' '.join([pd.DataFrame(A.inputs.iloc[0])['input_script_string'].iloc[0],pd.DataFrame(A.outputs.iloc[0])['output_script_string'].iloc[0]])
print('----------- redeem script (input) -------------')
for command in pd.DataFrame(A.inputs.iloc[0])['input_script_string'].iloc[0].split(' '):
    print(command)
print('----------- locking script (output) -----------')
for command in pd.DataFrame(A.outputs.iloc[0])['output_script_string'].iloc[0].split(' '):
    print(command)
from base58check import b58encode
from hashlib import sha256
from hashlib import new as hnew

def pubkey_to_hash(pubkey_string):
    hash_160=hnew('ripemd160')
    hash_160.update(sha256(bytes.fromhex(pubkey_string)).digest())
    return hash_160.hexdigest()

def address_from_hash(hash_string,pubkey=True):
    prefix='00' if pubkey else '05'
    PubKeyHash=bytes.fromhex(prefix+hash_string)
    checksum=sha256(sha256(PubKeyHash).digest()).digest()[:4]
    haa=PubKeyHash+checksum
    return b58encode(haa).decode('utf-8')

def address_from_pubkey(pubkey_string,pubkey=True):
    return address_from_hash(pubkey_to_hash(pubkey_string),pubkey)
pub_key=script.split(' ')[0][-67:-1]
hash_lock=script.split(' ')[4][-41:-1]
print('The hashed public key is {},\nwhich is not equal to {}.\nTherefore the script fails to execute and the transaction is denied.'.format(pubkey_to_hash(pub_key),hash_lock))
print('The addresses used are {} and {}, which are indeed different.'.format(address_from_pubkey(pub_key),address_from_hash(hash_lock)))
display_dict={'input_pubkey_base58':'input address','input_pubkey_base58_error':'key error', 'input_script_byte': 'script byte','input_script_string': 'script str', 'input_script_string_error':'script_error','input_sequence_number':'seq','output_pubkey_base58':'output address','output_pubkey_base58_error':'key error', 'output_script_byte': 'script byte','output_script_string': 'script str', 'output_script_string_error':'script_error'}
inputs=(pd.concat([pd.DataFrame(data=A.inputs[i]).assign(transaction_id=A.transaction_id[i],timestamp=A.timestamp[i]) for i in range(len(A))])
                .reset_index(drop=True))
msno.matrix(inputs.rename(columns=display_dict),figsize=(12,6));
outputs=(pd.concat([pd.DataFrame(data=A.outputs[i]).assign(transaction_id=A.transaction_id[i],timestamp=A.timestamp[i]) for i in range(len(A))])
                .reset_index(drop=True))
msno.matrix(outputs.rename(columns=display_dict),figsize=(12,6));
inputs=inputs.loc[(inputs.input_script_string_error.isnull()) & ~(inputs.input_pubkey_base58=='')]
outputs=outputs.loc[outputs.output_script_string_error.isnull()]
print('input address errors are the following:')
for x in inputs.input_pubkey_base58_error.unique():
    if x is not None:
        print(x)
print('\noutput address errors are the following:')
for x in outputs.output_pubkey_base58_error.unique():
    if x is not None:
        print(x)
P2PKH_input=re.compile(r'^PUSHDATA\([67]\d\).+PUSHDATA\((?:33|35|65)\)')
print('{0:.2f}% missing input addresses have a P2PKH type script.'.format(len(inputs[inputs.input_script_string.str.contains(P2PKH_input, na=False) & inputs.input_pubkey_base58.isnull()])/len(inputs[inputs.input_pubkey_base58.isnull()])*100))
print('{0:.2f}% of input addresses come from a P2PKH type script.'.format(len(inputs[inputs.input_script_string.str.contains(P2PKH_input, na=False) & ~inputs.input_pubkey_base58.isnull()])/len(inputs[~inputs.input_pubkey_base58.isnull()])*100))
P2PK_input=re.compile(r'^PUSHDATA\(7[1-2]\)\[\w+\]$')
print('{0:.2f}% missing input addresses have a P2PK type script.'.format(len(inputs[inputs.input_script_string.str.contains(P2PK_input, na=False) & inputs.input_pubkey_base58.isnull()])/len(inputs[inputs.input_pubkey_base58.isnull()])*100))
print('{0:.2f}% of input addresses come from a P2PK type script.'.format(len(inputs[inputs.input_script_string.str.contains(P2PK_input, na=False) & ~inputs.input_pubkey_base58.isnull()])/len(inputs[~inputs.input_pubkey_base58.isnull()])*100))
P2PKH_output=re.compile(r'^DUP HASH160 PUSHDATA\(20\)\[[a-f-0-9]{40}\] EQUALVERIFY CHECKSIG')
print('{0:.2f}% missing output addresses have a P2PKH type script.'.format(len(outputs[outputs.output_script_string.str.contains(P2PKH_output, na=False) & outputs.output_pubkey_base58.isnull()])/len(outputs[outputs.output_pubkey_base58.isnull()])*100))
print('{0:.2f}% of output addresses come from a P2PKH type script.'.format(len(outputs[outputs.output_script_string.str.contains(P2PKH_output, na=False) & ~outputs.output_pubkey_base58.isnull()])/len(outputs[~outputs.output_pubkey_base58.isnull()])*100))
print('/nand/n')
P2PK_output=re.compile(r'^PUSHDATA\(3[35]\)\[\w+\] CHECKSIG$')
print('{0:.2f}% missing output addresses have a P2PK type script.'.format(len(outputs[outputs.output_script_string.str.contains(P2PK_output, na=False) & outputs.output_pubkey_base58.isnull()])/len(outputs[outputs.output_pubkey_base58.isnull()])*100))
print('{0:.2f}% of output addresses come from a P2PK type script.'.format(len(outputs[outputs.output_script_string.str.contains(P2PK_output, na=False) & ~outputs.output_pubkey_base58.isnull()])/len(outputs[~outputs.output_pubkey_base58.isnull()])*100))
print('/nwhile/n')
SCRIPT_output=re.compile(r'HASH160 PUSHDATA\(20\)\[[a-f-0-9]{40}\]')
print('{0:.2f}% missing output addresses have a script with \'HASH160 PUSHDATA(20)\'.'.format(len(outputs[outputs.output_script_string.str.contains(SCRIPT_output, na=False) & outputs.output_pubkey_base58.isnull()])/len(outputs[outputs.output_pubkey_base58.isnull()])*100))
print('{0:.2f}% of output addresses have a script with \'HASH160 PUSHDATA(20)\'.'.format(len(outputs[outputs.output_script_string.str.contains(SCRIPT_output, na=False) & ~outputs.output_pubkey_base58.isnull()])/len(outputs[~outputs.output_pubkey_base58.isnull()])*100))
_,axes=plt.subplots(ncols=2,figsize=(16,6))
inputs.input_pubkey_base58.dropna().str.extract(r'^(\d)',expand=False).value_counts().plot.bar(title='distribution of first digit of input addresses',ax=axes[0]);
outputs.output_pubkey_base58.dropna().str.extract(r'^(\d)',expand=False).value_counts().plot.bar(title='distribution of first digit of output addresses',ax=axes[1]);
print('----------- redeem script (input) -------------')
print('0\nPUSHDATA(71)[sig1]\nPUSHDATA(71)[sig2]')
print('----------- locking script (output) -----------')
print('2\nPUSHDATA(33)[pubkey1]\nPUSHDATA(33)[pubkey2]\nPUSHDATA(33)[pubkey3]\n3\nCHECKMULTISIG')
MULTISIG_input=re.compile(r'^0')
print('There are {} records where an input public key is missing and the script is of the MULTISIG type.'.format(len(inputs[inputs.input_script_string.str.contains(MULTISIG_input, na=False) & inputs.input_pubkey_base58.isnull()])))
print('It corresponds to {0: .1f}% of the missing addresses'.format(inputs[inputs.input_pubkey_base58.isnull()].input_script_string.str.contains(MULTISIG_input, na=False).mean()*100))
inputs[inputs.input_pubkey_base58.isnull()].input_script_string.str.extractall(r'(PUSHDATA[\(\d]\d*\)*)').unstack(level=1).drop_duplicates()[0].sort_values(by=[0,1,2,3]).reset_index(drop=True).fillna('')
print('----------- redeem script (input) -------------')
print('0\nPUSHDATA(71)[sig1]\nPUSHDATA(71)[sig2]')
print('PUSHDATA1[multisig_script]')
print('----------- locking script (output) -----------')
print('HASH160\nPUSHDATA(20)[script_hash]\nEQUAL')
print('----------- redeem script (input) -------------')
print('preload_data\nPUSHDATA1[script]')
print('----------- locking script (output) -----------')
print('HASH160\nPUSHDATA(20)[script_hash]\nEQUAL')
inputs[(inputs.input_pubkey_base58.isnull())&(inputs.input_script_string.str.contains(r'PUSHDATA[124]'))]=(inputs[(inputs.input_pubkey_base58.isnull())&(inputs.input_script_string.str.contains(r'PUSHDATA[124]'))]
 .assign(input_pubkey_base58=lambda x: x.input_script_string.str.extract(r'PUSHDATA[124]\[(\w+)\]').iloc[:,0].transform(address_from_pubkey,pubkey=False))
)
_,axes=plt.subplots(ncols=2,figsize=(16,6))
inputs.input_pubkey_base58.dropna().str.extract(r'^(\d)',expand=False).value_counts().plot.bar(title='distribution of first digit of input addresses',ax=axes[0]);
outputs.output_pubkey_base58.dropna().str.extract(r'^(\d)',expand=False).value_counts().plot.bar(title='distribution of first digit of output addresses',ax=axes[1]);
msno.matrix(inputs.rename(columns=display_dict),figsize=(12,6));
SEGWIT_output=re.compile(r'^0\[\] PUSHDATA\(20|32\)\[\w+\]$')
print('{0:.2f}% missing output addresses have a SEGWIT type script.'.format(len(outputs[outputs.output_script_string.str.contains(SEGWIT_output, na=False) & outputs.output_pubkey_base58.isnull()])/len(outputs[outputs.output_pubkey_base58.isnull()])*100))
print('{0:.2f}% of output addresses come from a SEGWIT type script.'.format(len(outputs[outputs.output_script_string.str.contains(SEGWIT_output, na=False) & ~outputs.output_pubkey_base58.isnull()])/len(outputs[~outputs.output_pubkey_base58.isnull()])*100))

inputs=(pd.concat([pd.DataFrame(data=A.inputs[i])[['input_pubkey_base58']].assign(transaction_id=A.transaction_id[i],timestamp=A.timestamp[i],kind='input') for i in range(n)])
                .rename(columns={'input_pubkey_base58':'address'}))
outputs=(pd.concat([pd.DataFrame(data=A.outputs[i])[['output_pubkey_base58','output_satoshis']].assign(transaction_id=A.transaction_id[i],timestamp=A.timestamp[i],kind='output') for i in range(n)])
                .rename(columns={'output_pubkey_base58':'address'}))
all_transactions=pd.concat([inputs,outputs],sort=False).reset_index(drop=True)
all_transactions['date']=pd.to_datetime(all_transactions.timestamp,unit='ms')
last_transaction=A.transaction_id[0]

all_transactions[all_transactions.transaction_id==last_transaction]
SBtx=(all_transactions.drop_duplicates(subset=['address','kind','transaction_id'])
                      .loc[lambda x: x.duplicated(subset=['address','transaction_id'],keep=False),'transaction_id']
                      .unique()
                     )
fig,axes=plt.subplots(ncols=2,figsize=(20,6))
(all_transactions
     .loc[:,['transaction_id']]
     .drop_duplicates()
     .assign(is_regular=lambda x: x.transaction_id.isin(SBtx))
     .is_regular
     .value_counts()
     .plot.bar(title='Total transactions by type',ax=axes[0])
);
axes[0].set_xticklabels(['non-SBtx','SBtx'])
axes[0].tick_params(rotation=0)
(all_transactions
     .set_index('date')
     .loc[:,['transaction_id']]
     .drop_duplicates()
     .assign(is_regular=lambda x: x.transaction_id.isin(SBtx))
     .is_regular
     .astype(np.int8)
     .resample('M')
     .sum()
     .plot(ax=axes[1],title='Volume of transactions by type',color='orange')
);
(all_transactions
     .set_index('date')
     .loc[:,['transaction_id']]
     .drop_duplicates()
     .assign(is_regular=lambda x: ~x.transaction_id.isin(SBtx))
     .is_regular
     .astype(np.int8)
     .resample('M')
     .sum()
     .plot(ax=axes[1])
);
disposable_addresses=(all_transactions
                    .drop_duplicates(subset=['address','transaction_id'])
                    .address
                    .value_counts()
                    .reset_index()
                    .loc[lambda x :x.address==1,'index']
                    .values)
disposable_transactions=(all_transactions
                         .loc[all_transactions['address']
                         .isin(disposable_addresses),'transaction_id']
                         .drop_duplicates()
                         .values)
use_count=(all_transactions
     .drop_duplicates(subset=['address','transaction_id'])
     .address
     .value_counts()
     .value_counts()
)
fig,axes=plt.subplots(ncols=2,figsize=(20,6))
squarify.plot(sizes=use_count,label=use_count.index,ax=axes[0]);
axes[0].set_title('Distribution of addresses by number of transactions')
axes[0].axis('off');
(all_transactions
     .set_index('date')
     .loc[:,['transaction_id']]
     .drop_duplicates()
     .assign(is_disposable=lambda x: ~x.transaction_id.isin(disposable_transactions))
     .is_disposable
     .astype(np.int8)
     .resample('M')
     .sum()
     .plot(ax=axes[1])
);
(all_transactions
     .set_index('date')
     .loc[:,['transaction_id']]
     .drop_duplicates()
     .assign(is_disposable=lambda x: x.transaction_id.isin(disposable_transactions))
     .is_disposable
     .astype(np.int8)
     .resample('M')
     .sum()
     .plot(ax=axes[1],color='orange')
);
axes[1].set_title('transactions containing single use addresses (orange) vs others');