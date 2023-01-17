!pip install tldextract
!pip install flask-ngrok
!pip install flask_cors
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import itertools
import tldextract
import numpy as np
import json
from flask import Flask, jsonify, request
from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin
#loading pickle files

subdomain_tokens = pickle.load(open( "../input/1dcnnurlclassifierpkl/subdomain.pkl", "rb" ))
domain_tokens = pickle.load(open( "../input/1dcnnurlclassifierpkl/domain.pkl", "rb" ))
domain_suffix_tokens = pickle.load(open( "../input/1dcnnurlclassifierpkl/domain_suffix.pkl", "rb" ))
tokenizer = pickle.load(open( "../input/1dcnnurlclassifierpkl/tokenizer.pkl", "rb" ))
sequence_length = 162
n_char = 177
unique_subdomain = 23345
unique_domain = 98290
unique_domain_suffix = 673
def encode_label(label_index, data):
    try:
        return label_index[data]
    except:
        return label_index['<unknown>']
def extract_url(url):
    train_seq = tokenizer.texts_to_sequences(url)
    train_seq = list(itertools.chain(*train_seq))
    train_seq = pad_sequences([train_seq], padding='post', maxlen=sequence_length)
    train_seq = np.divide(train_seq, n_char).tolist()
    
    # extracting subdomain, domain, domain_suffix from url using tldextract
    subdomain, domain, domain_suffix = tldextract.extract(url)

    tokenized_subdomain = [encode_label(subdomain_tokens,subdomain)]
    tokenized_domain = [encode_label(domain_tokens,domain)]
    tokenized_domain_suffix = [encode_label(domain_suffix_tokens,domain_suffix)]
    
    # normalizing
    tokenized_subdomain = np.divide(tokenized_subdomain,unique_subdomain).tolist()
    tokenized_domain = np.divide(tokenized_domain,unique_domain).tolist()
    tokenized_domain_suffix = np.divide(tokenized_domain_suffix,unique_domain_suffix).tolist()
    
    return [train_seq[0],tokenized_subdomain,tokenized_domain,tokenized_domain_suffix]
app = Flask(__name__)
run_with_ngrok(app)
cors = CORS(app)

@app.route('/parseurl',methods=["GET"])
@cross_origin()
def hello():
    url = request.args.get('url')
    print(url)
    data = extract_url(url)
    return jsonify({"data":data})

app.run()