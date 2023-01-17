%%writefile convert_vocab.py
import argparse
import os
from pathlib import Path
from time import sleep

import re
import json

import translators as ts

not_en = re.compile(r'[^a-z^A-Z]')


def vocab_convertor(path_vocab, tgt_lang, path_bilingual_dict):

    flag = False
    fname, ext = path_vocab.split('.')
    path_new_vocab = fname + '_' + tgt_lang + '.json'
    cache_path = 'translated-' + tgt_lang  + '.log'
    
   # translated = open('translated-' + tgt_lang  + '.log', 'a', encoding='UTF-8')
    
    with open(path_vocab, encoding='UTF-8') as json_file:
        vocab = json.load(json_file)
    
    translated_dict = {}
    bilingual_dict = {}
    new_vocab = {}
    
    
    with open(path_bilingual_dict, encoding='UTF-8') as f:
        try:
            for line in f:
                print(line)
                src, tgt = line.split(' ')
                if not src in bilingual_dict:
                    bilingual_dict[src] = tgt.replace('\n', '')
        except:
            for line in f:
                print(line)
                src, tgt = line.split('\t')
                if not src in bilingual_dict:
                    bilingual_dict[src] = tgt.replace('\n', '')
    
    if not os.path.exists(cache_path):
        Path(cache_path).touch()

    with open(cache_path, encoding='UTF-8') as f:
        for line in f:
            src_tgt__ = line.split(' ')
            src = src_tgt__[0]
            tgt = src_tgt__[1]
            translated_dict[src] = tgt.replace('\n', '')
        
        
    for i, (k, v) in enumerate(vocab.items()):

        with open('translated-' + tgt_lang  + '.log', 'a', encoding='UTF-8') as translated:
            print(k, v)
        
            _k = k.split('\u0120')
            k_old = _k[-1]
            k = None
        
        
            if bool(not_en.search(k_old)) | (v == 220) | (1==len(k_old)):
                k = k_old
                
            else:
            
                if k_old in bilingual_dict:
                    k = bilingual_dict[k_old]
                elif k_old in translated_dict:
                    k = translated_dict[k_old]
                else:
                    k_ts = ts.google(k_old, 'en', tgt_lang)
                
                    translated.write(k_old+' '+k_ts+'\n')
                    k = k_ts
                    sleep(2)
                    
            
            if 1 == len(_k):
                new_vocab[k] = v
            else:
                new_vocab['\u0120'+k] = v
            
        
            
    with open(path_new_vocab,"w", encoding='utf-8') as jsonfile:
        json.dump(new_vocab,jsonfile,ensure_ascii=False)
    
    #"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("path_vocab", type=str)
    parser.add_argument("tgt_lang", type=str)
    parser.add_argument("path_bilingual_dict", type=str)
    args = parser.parse_args()

    vocab_convertor(args.path_vocab, args.tgt_lang, args.path_bilingual_dict)