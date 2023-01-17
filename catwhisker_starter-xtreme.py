!pip install transformers==3.0.2 --quiet

!pip install nlp --quiet
TASK= {

    'udpos': {},

    'panx': {},

    'xnli': {},

    'pawsx': {},

    'xquad': {},

    'mlqa': {},

    'tydiqa': {},

    'bucc2018': {},

    'tatoeba': {}

}

TASK['udpos']['train'] = ('xtreme', 'udpos.English','train')

TASK['udpos']['epochs'] = 10

TASK['udpos']['max seq length'] = 128

TASK['udpos']['learning rate'] = 2e-5

TASK['udpos']['weight decay'] = 0

TASK['udpos']['warmup step'] = 0

TASK['udpos']['validation'] = ('xtreme', 'udpos.English','validation')

TASK['udpos']['test'] = {}

TASK['udpos']['test']['en'] = ('xtreme', 'udpos.English','test')

TASK['udpos']['test']['af'] = ('xtreme', 'udpos.Afrikaans','test')

TASK['udpos']['test']['ar'] = ('xtreme', 'udpos.Arabic','test')

TASK['udpos']['test']['eu'] = ('xtreme', 'udpos.Basque','test')

TASK['udpos']['test']['bg'] = ('xtreme', 'udpos.Bulgarian','test')

TASK['udpos']['test']['nl'] = ('xtreme', 'udpos.Dutch','test')

TASK['udpos']['test']['et'] = ('xtreme', 'udpos.Estonian','test')

TASK['udpos']['test']['fi'] = ('xtreme', 'udpos.Finnish','test')

TASK['udpos']['test']['fr'] = ('xtreme', 'udpos.French','test')

TASK['udpos']['test']['de'] = ('xtreme', 'udpos.German','test')

TASK['udpos']['test']['el'] = ('xtreme', 'udpos.Greek','test')

TASK['udpos']['test']['he'] = ('xtreme', 'udpos.Hebrew','test')

TASK['udpos']['test']['hi'] = ('xtreme', 'udpos.Hindi','test')

TASK['udpos']['test']['hu'] = ('xtreme', 'udpos.Hungarian','test')

TASK['udpos']['test']['id'] = ('xtreme', 'udpos.Indonesian','test')

TASK['udpos']['test']['it'] = ('xtreme', 'udpos.Italian','test')

TASK['udpos']['test']['ja'] = ('xtreme', 'udpos.Japanese','test')

TASK['udpos']['test']['kk'] = ('xtreme', 'udpos.Kazakh','test')

TASK['udpos']['test']['ko'] = ('xtreme', 'udpos.Korean','test')

TASK['udpos']['test']['zh'] = ('xtreme', 'udpos.Chinese','test')

TASK['udpos']['test']['mr'] = ('xtreme', 'udpos.Marathi','test')

TASK['udpos']['test']['fa'] = ('xtreme', 'udpos.Persian','test')

TASK['udpos']['test']['pt'] = ('xtreme', 'udpos.Portuguese','test')

TASK['udpos']['test']['ru'] = ('xtreme', 'udpos.Russian','test')

TASK['udpos']['test']['es'] = ('xtreme', 'udpos.Spanish','test')

TASK['udpos']['test']['tl'] = ('xtreme', 'udpos.Tagalog','test')

TASK['udpos']['test']['ta'] = ('xtreme', 'udpos.Tamil','test')

TASK['udpos']['test']['te'] = ('xtreme', 'udpos.Telugu','test')

TASK['udpos']['test']['th'] = ('xtreme', 'udpos.Thai','test')

TASK['udpos']['test']['tr'] = ('xtreme', 'udpos.Turkish','test')

TASK['udpos']['test']['ur'] = ('xtreme', 'udpos.Urdu','test')

TASK['udpos']['test']['vi'] = ('xtreme', 'udpos.Vietnamese','test')

TASK['udpos']['test']['yo'] = ('xtreme', 'udpos.Yoruba','test')







TASK['panx']['train'] = ('xtreme', 'PAN-X.en','train')

TASK['panx']['epochs'] = 10

TASK['panx']['max seq length'] = 128

TASK['panx']['learning rate'] = 2e-5

TASK['panx']['warmup step'] = 0

TASK['panx']['weight decay'] = 0

TASK['panx']['validation'] = ('xtreme', 'PAN-X.en','validation')

TASK['panx']['test'] = {}

TASK['panx']['test']['en'] = ('xtreme', 'PAN-X.en','test')

TASK['panx']['test']['af'] = ('xtreme', 'PAN-X.af','test')

TASK['panx']['test']['ar'] = ('xtreme', 'PAN-X.ar','test')

TASK['panx']['test']['bg'] = ('xtreme', 'PAN-X.bg','test')

TASK['panx']['test']['bn'] = ('xtreme', 'PAN-X.bn','test')

TASK['panx']['test']['de'] = ('xtreme', 'PAN-X.de','test')

TASK['panx']['test']['el'] = ('xtreme', 'PAN-X.el','test')

TASK['panx']['test']['en'] = ('xtreme', 'PAN-X.en','test')

TASK['panx']['test']['es'] = ('xtreme', 'PAN-X.es','test')

TASK['panx']['test']['et'] = ('xtreme', 'PAN-X.et','test')

TASK['panx']['test']['eu'] = ('xtreme', 'PAN-X.eu','test')

TASK['panx']['test']['fa'] = ('xtreme', 'PAN-X.fa','test')

TASK['panx']['test']['fi'] = ('xtreme', 'PAN-X.fi','test')

TASK['panx']['test']['fr'] = ('xtreme', 'PAN-X.fr','test')

TASK['panx']['test']['he'] = ('xtreme', 'PAN-X.he','test')

TASK['panx']['test']['hi'] = ('xtreme', 'PAN-X.hi','test')

TASK['panx']['test']['hu'] = ('xtreme', 'PAN-X.hu','test')

TASK['panx']['test']['id'] = ('xtreme', 'PAN-X.id','test')

TASK['panx']['test']['it'] = ('xtreme', 'PAN-X.it','test')

TASK['panx']['test']['ja'] = ('xtreme', 'PAN-X.ja','test')

TASK['panx']['test']['jv'] = ('xtreme', 'PAN-X.jv','test')

TASK['panx']['test']['ka'] = ('xtreme', 'PAN-X.ka','test')

TASK['panx']['test']['kk'] = ('xtreme', 'PAN-X.kk','test')

TASK['panx']['test']['ko'] = ('xtreme', 'PAN-X.ko','test')

TASK['panx']['test']['ml'] = ('xtreme', 'PAN-X.ml','test')

TASK['panx']['test']['mr'] = ('xtreme', 'PAN-X.mr','test')

TASK['panx']['test']['ms'] = ('xtreme', 'PAN-X.ms','test')

TASK['panx']['test']['my'] = ('xtreme', 'PAN-X.my','test')

TASK['panx']['test']['nl'] = ('xtreme', 'PAN-X.nl','test')

TASK['panx']['test']['pt'] = ('xtreme', 'PAN-X.pt','test')

TASK['panx']['test']['ru'] = ('xtreme', 'PAN-X.ru','test')

TASK['panx']['test']['sw'] = ('xtreme', 'PAN-X.sw','test')

TASK['panx']['test']['ta'] = ('xtreme', 'PAN-X.ta','test')

TASK['panx']['test']['te'] = ('xtreme', 'PAN-X.te','test')

TASK['panx']['test']['th'] = ('xtreme', 'PAN-X.th','test')

TASK['panx']['test']['tl'] = ('xtreme', 'PAN-X.tl','test')

TASK['panx']['test']['tr'] = ('xtreme', 'PAN-X.tr','test')

TASK['panx']['test']['ur'] = ('xtreme', 'PAN-X.ur','test')

TASK['panx']['test']['vi'] = ('xtreme', 'PAN-X.vi','test')

TASK['panx']['test']['yo'] = ('xtreme', 'PAN-X.yo','test')

TASK['panx']['test']['zh'] = ('xtreme', 'PAN-X.zh','test')



TASK['xnli']['train'] = ("multi_nli", None, 'train')

TASK['xnli']['epochs'] = 5

TASK['xnli']['max seq length'] = 128

TASK['xnli']['learning rate'] = 2e-5

TASK['xnli']['warmup step'] = 0

TASK['xnli']['weight decay'] = 0

TASK['xnli']['validation_matched'] = ('multi_nli', None,'validation_matched')

TASK['xnli']['validation_mismatched'] = ('multi_nli', 'None','validation_mismatched')

TASK['xnli']['test'] = ('xtreme', 'XNLI','test')



TASK['pawsx']['train'] = ('xtreme', 'PAWS-X.en','train')

TASK['pawsx']['epochs'] = 5

TASK['pawsx']['max seq length'] = 128

TASK['pawsx']['learning rate'] = 2e-5

TASK['pawsx']['warmup step'] = 0

TASK['pawsx']['weight decay'] = 0

TASK['pawsx']['validation'] = ('xtreme', 'PAWS-X.en','validation')

TASK['pawsx']['test'] = {}

TASK['pawsx']['test']['en'] = ('xtreme', 'PAWS-X.en','test')

TASK['pawsx']['test']['es'] = ('xtreme', 'PAWS-X.es','test')

TASK['pawsx']['test']['de'] = ('xtreme', 'PAWS-X.de','test')

TASK['pawsx']['test']['fr'] = ('xtreme', 'PAWS-X.fr','test')

TASK['pawsx']['test']['ja'] = ('xtreme', 'PAWS-X.ja','test')

TASK['pawsx']['test']['ko'] = ('xtreme', 'PAWS-X.ko','test')

TASK['pawsx']['test']['zh'] = ('xtreme', 'PAWS-X.zh','test')



TASK['xquad']['train'] = ('xtreme', 'SQuAD','train')

TASK['xquad']['epochs'] = 2

TASK['xquad']['max seq length'] = 384

TASK['xquad']['learning rate'] = 3e-5

TASK['xquad']['warmup step'] = 500

TASK['xquad']['weight decay'] = 0.0001 

TASK['xquad']['validation'] = ('xtreme', 'SQuAD','validation')

TASK['xquad']['test'] = {} 

TASK['xquad']['test']['ar'] =  ('xtreme', 'XQuAD.ar','validation')

TASK['xquad']['test']['de'] =  ('xtreme', 'XQuAD.de','validation')

TASK['xquad']['test']['vi'] =  ('xtreme', 'XQuAD.vi','validation')

TASK['xquad']['test']['zh'] =  ('xtreme', 'XQuAD.zh','validation')

TASK['xquad']['test']['en'] =  ('xtreme', 'XQuAD.en','validation')

TASK['xquad']['test']['es'] =  ('xtreme', 'XQuAD.es','validation')

TASK['xquad']['test']['hi'] =  ('xtreme', 'XQuAD.hi','validation')

TASK['xquad']['test']['el'] =  ('xtreme', 'XQuAD.el','validation')

TASK['xquad']['test']['ru'] =  ('xtreme', 'XQuAD.ru','validation')

TASK['xquad']['test']['th'] =  ('xtreme', 'XQuAD.th','validation')

TASK['xquad']['test']['tr'] =  ('xtreme', 'XQuAD.tr','validation')



"""

Xtreme ignore subset of mlqa which if context & question with different langauge

https://github.com/google-research/xtreme/blob/master/scripts/eval_qa.sh

echo "MLQA"

for lang in en es de ar hi vi zh; do

 echo -n "  $lang "

 TEST_FILE=${MLQA_DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json

 PRED_FILE=${MLQA_PRED_DIR}/predictions_${lang}_.json

 python "${EVAL_MLQA}" "${TEST_FILE}" "${PRED_FILE}" ${lang}

done

"""

TASK['mlqa']['train'] = ('xtreme', 'SQuAD','train')

TASK['mlqa']['epochs'] = 2

TASK['mlqa']['max seq length'] = 384

TASK['mlqa']['learning rate'] = 3e-5

TASK['mlqa']['warmup step'] = 500

TASK['mlqa']['weight decay'] = 0.0001 

TASK['mlqa']['validation'] = ('xtreme', 'SQuAD','validation')

TASK['mlqa']['test'] = {}

TASK['mlqa']['test']['ar'] =  ('xtreme', 'MLQA.ar.ar','test')

TASK['mlqa']['test']['de'] =  ('xtreme', 'MLQA.de.de','test')

TASK['mlqa']['test']['vi'] =  ('xtreme', 'MLQA.vi.vi','test')

TASK['mlqa']['test']['zh'] =  ('xtreme', 'MLQA.zh.zh','test')

TASK['mlqa']['test']['en'] =  ('xtreme', 'MLQA.en.en','test')

TASK['mlqa']['test']['es'] =  ('xtreme', 'MLQA.es.es','test')

TASK['mlqa']['test']['hi'] =  ('xtreme', 'MLQA.hi.hi','test')



TASK['tydiqa']['train'] = ('xtreme', 'tydiqa','train')

TASK['tydiqa']['epochs'] = 2

TASK['tydiqa']['max seq length'] = 384

TASK['tydiqa']['learning rate'] = 3e-5

TASK['tydiqa']['warmup step'] = 500

TASK['tydiqa']['weight decay'] = 0.0001 

TASK['tydiqa']['test'] =  ('xtreme', 'tydiqa','validatoin')



TASK['bucc2018']['src'] = {}

TASK['bucc2018']['max seq length'] = 512

TASK['bucc2018']['src']['de'] =  ('xtreme', 'bucc18.de','validation')

TASK['bucc2018']['src']['fr'] =  ('xtreme', 'bucc18.fr','validation')

TASK['bucc2018']['src']['zh'] =  ('xtreme', 'bucc18.zh','validation')

TASK['bucc2018']['src']['ru'] =  ('xtreme', 'bucc18.ru','validation')



TASK['tatoeba']['src'] = {}

TASK['tatoeba']['max seq length'] = 512

TASK['tatoeba']['src']['afr'] =  ('xtreme', 'tatoeba.afr','validation')

TASK['tatoeba']['src']['ara'] =  ('xtreme', 'tatoeba.ara','validation')

TASK['tatoeba']['src']['ben'] =  ('xtreme', 'tatoeba.ben','validation')

TASK['tatoeba']['src']['bul'] =  ('xtreme', 'tatoeba.bul','validation')

TASK['tatoeba']['src']['deu'] =  ('xtreme', 'tatoeba.deu','validation')

TASK['tatoeba']['src']['cmn'] =  ('xtreme', 'tatoeba.cmn','validation')

TASK['tatoeba']['src']['ell'] =  ('xtreme', 'tatoeba.ell','validation')

TASK['tatoeba']['src']['est'] =  ('xtreme', 'tatoeba.est','validation')

TASK['tatoeba']['src']['eus'] =  ('xtreme', 'tatoeba.eus','validation')

TASK['tatoeba']['src']['fin'] =  ('xtreme', 'tatoeba.fin','validation')

TASK['tatoeba']['src']['fra'] =  ('xtreme', 'tatoeba.fra','validation')

TASK['tatoeba']['src']['heb'] =  ('xtreme', 'tatoeba.heb','validation')

TASK['tatoeba']['src']['hin'] =  ('xtreme', 'tatoeba.hin','validation')

TASK['tatoeba']['src']['hun'] =  ('xtreme', 'tatoeba.hun','validation')

TASK['tatoeba']['src']['ind'] =  ('xtreme', 'tatoeba.ind','validation')

TASK['tatoeba']['src']['ita'] =  ('xtreme', 'tatoeba.ita','validation')

TASK['tatoeba']['src']['jav'] =  ('xtreme', 'tatoeba.jav','validation')

TASK['tatoeba']['src']['jpn'] =  ('xtreme', 'tatoeba.jpn','validation')

TASK['tatoeba']['src']['kat'] =  ('xtreme', 'tatoeba.kat','validation')

TASK['tatoeba']['src']['kaz'] =  ('xtreme', 'tatoeba.kaz','validation')

TASK['tatoeba']['src']['kor'] =  ('xtreme', 'tatoeba.kor','validation')

TASK['tatoeba']['src']['mal'] =  ('xtreme', 'tatoeba.mal','validation')

TASK['tatoeba']['src']['mar'] =  ('xtreme', 'tatoeba.mar','validation')

TASK['tatoeba']['src']['nld'] =  ('xtreme', 'tatoeba.nld','validation')

TASK['tatoeba']['src']['pes'] =  ('xtreme', 'tatoeba.pes','validation')

TASK['tatoeba']['src']['por'] =  ('xtreme', 'tatoeba.por','validation')

TASK['tatoeba']['src']['rus'] =  ('xtreme', 'tatoeba.rus','validation')

TASK['tatoeba']['src']['spa'] =  ('xtreme', 'tatoeba.spa','validation')

TASK['tatoeba']['src']['swh'] =  ('xtreme', 'tatoeba.swh','validation')

TASK['tatoeba']['src']['tam'] =  ('xtreme', 'tatoeba.tam','validation')

TASK['tatoeba']['src']['tgl'] =  ('xtreme', 'tatoeba.tgl','validation')

TASK['tatoeba']['src']['tha'] =  ('xtreme', 'tatoeba.tha','validation')

TASK['tatoeba']['src']['tur'] =  ('xtreme', 'tatoeba.tur','validation')

TASK['tatoeba']['src']['urd'] =  ('xtreme', 'tatoeba.urd','validation')

TASK['tatoeba']['src']['vie'] =  ('xtreme', 'tatoeba.vie','validation')



#copied from https://github.com/google-research/xtreme/blob/master/third_party/run_retrieval.py

lang3_dict = {'ara':'ar', 'heb':'he', 'vie':'vi', 'ind':'id',

    'jav':'jv', 'tgl':'tl', 'eus':'eu', 'mal':'ml', 'tam':'ta',

    'tel':'te', 'afr':'af', 'nld':'nl', 'eng':'en', 'deu':'de',

    'ell':'el', 'ben':'bn', 'hin':'hi', 'mar':'mr', 'urd':'ur',

    'tam':'ta', 'fra':'fr', 'ita':'it', 'por':'pt', 'spa':'es',

    'bul':'bg', 'rus':'ru', 'jpn':'ja', 'kat':'ka', 'kor':'ko',

    'tha':'th', 'swh':'sw', 'cmn':'zh', 'kaz':'kk', 'tur':'tr',

    'est':'et', 'fin':'fi', 'hun':'hu', 'pes':'fa'}



TASK['tatoeba']['src'] ={code2: TASK['tatoeba']['src'].get(code3) for (code3, code2) in lang3_dict.items()}





TASK2LANGS = {

  "pawsx": "de,en,es,fr,ja,ko,zh".split(","),

  "xnli": "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh".split(","),

  "panx": "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu".split(","),

  "udpos": "af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh".split(","),

  "bucc2018": "de,fr,ru,zh".split(","),

  "tatoeba": "ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(","),

  "xquad": "en,es,de,el,ru,tr,ar,vi,th,zh,hi".split(","),

  "mlqa": "en,es,de,ar,hi,vi,zh".split(","),

  "tydiqa": "en,ar,bn,fi,id,ko,ru,sw,te".split(","),

}

keywords = ['epochs','learning rate','warmup step', 'weight decay']



assert TASK2LANGS.keys() == TASK.keys()

for task in TASK:

  if 'max seq length' not in TASK[task]:

    print(task + ': max seq length not found')

  if task == 'bucc2018' or task == 'tatoeba':

    for code in TASK2LANGS[task]:

      if code not in TASK[task]['src']:

        print(task + ': test split for '+ code +' not found')

  else:

    for word in keywords:

      if word not in TASK[task]:

        print(task + ': ' + word +' not found')

    if task == 'xnli' or task == 'tydiqa':

      continue

    for code in TASK2LANGS[task]:

      if code not in TASK[task]['test']:

        print(task + ': test split for '+ code +' not found')



def check_dataset(ds, task):

    pass



print('sanity check done!\nlanguage code check is not done for xnli or tydiqa\nplease specify data dir for panx')

import os, shutil



cache_path = '/root/.cache/huggingface/datasets/'

def pan_readonly2cache():

    src = '/kaggle/input/pan-x'

    dst = '/root/.cache/huggingface/datasets/xtreme'

    for item in os.listdir(src):

        s = os.path.join(src, item)

        d = os.path.join(dst, item)

        if os.path.isdir(s):

            shutil.copytree(s, d, False, None)

        else:

            shutil.copy2(s, d)
pan_readonly2cache()
from nlp import load_dataset

for task in TASK:

    try:

        set_name, subset_name, split = TASK[task]['train']

        load_dataset(set_name, subset_name)[split]

    except:

        pass
from nlp import load_dataset

for task in TASK:

    try:

        set_name, subset_name, split = TASK[task]['train']    

        print('training data for ' + task +' : '+str(len(load_dataset(set_name, subset_name)[split])))

    except:

        pass