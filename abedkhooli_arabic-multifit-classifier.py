# This sample notebook uses the Arabic MULTIFiT language model based on Arabic Wikipedia and 

# a few classic books. The model was used to fine-tune and train a small corpus of:



#---------- Case 1: text type (genre) classification --------------------

# Holy Qur'aan text

# Hadith from Bukhari

# Poetry (Mutannabi = Shawqi)



# The total number of records in the training set was around 39k. You can try the resulting classifier 

# in the last cell by providing your own test text (model can't tell if not one of the 3 categories above)



#----------- Case 2: hotel reviews classification --------------------

# based on the HARD dataset https://github.com/elnagara/HARD-Arabic-Dataset by

# Elnagar A., Khalifa Y.S., Einea A. (2018) - https://doi.org/10.1007/978-3-319-67056-0_3 

# the unbalanced dataset was cleaned keeping entries that are mainly Arabic then removing all non-Arabic 

# chars from those (only positive (4,5) and negative (1,2) reviews kept for a total of 325473). The model 

# achieved 97.4% F1. A copy of the cleaned dataset is available in this kernel.

# note: did not check for very short texts (after removing foreign chars, including emojis) - 955 records have only 2 words 



#Important Note: the model does not take diacritics or kashida. Remove these as well as non-Arabic chars first.
# in case fastai breaks, uninstall and install v 1.0.57

#!pip uninstall fastai --y

#!pip install ninja #ninja is already in the docker

!pip install sentencepiece

#!pip install fastai==1.0.57  # restart runtime then if done after fastai is imported
from fastai.text import *



import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "on



%reload_ext autoreload

%autoreload 2

%matplotlib inline
#!mkdir -p /content/models/ 

# need to move vocab and model here for fastai to work

!mkdir -p /root/.fastai/data/arwiki/corpus2_100/tmp/ 

!cp '/kaggle/input/arabicmf/spm.model' /root/.fastai/data/arwiki/corpus2_100/tmp/spm.model
path_ds = '/kaggle/input/arabicmf/'

bs = 18
%%time

data_clas = load_data(path_ds, f'ar_textlist_class_qhp_sp15_multifit', bs=bs, num_workers=1);



config = awd_lstm_clas_config.copy()

config['qrnn'] = True

config['n_hid'] = 1550 #default 1152

config['n_layers'] = 4 #default 3
learn_c = text_classifier_learner(data_clas, AWD_LSTM, config=config, pretrained=False)

learn_c.load(f'/kaggle/input/arabicmf/ar_clas_qhp_sp15_multifit', purge=False);
# predict random stuff (text must be in clean format - no diacritics or kashida) 



# here are three examples

test_text =  "عن ابن مسعود قال: قرأت على رسول الله من سورة النساء" # Hadith (category 1)

pred = learn_c.predict(test_text)

print(pred)



test_text = "لا تنه عن خلق وتأتي مثله عار عليك إذا فعلت عظيم" # Poetry (cataegory 2)

pred = learn_c.predict(test_text)

print(pred)



test_text ="أذن للذين يقاتلون بأنهم ظلموا وإن الله على نصرهم لقدير" # Qur'an (category 0)

pred = learn_c.predict(test_text)

print(pred)
#-------------------------- reviews 
%%time

path_ds = '/kaggle/input/arabicmf/'

bs = 18

data_clas_rev = load_data(path_ds, f'ar_textlist_class_hard_sp15_multifit', bs=bs, num_workers=1);



config = awd_lstm_clas_config.copy()

config['qrnn'] = True

config['n_hid'] = 1550 #default 1152

config['n_layers'] = 4 #default 3
learn_c_rev = text_classifier_learner(data_clas_rev, AWD_LSTM, config=config, pretrained=False)

learn_c_rev.load(f'/kaggle/input/arabicmf/ar_clas_hard_sp15_multifit', purge=False);
# some examples (not from dataset)

# positive

test_text_rev =  "كان المكان نظيفا والطعام جيدا. أوصي به للأصدقاء." #  (category 1)

pred_rev = learn_c_rev.predict(test_text_rev)

print(pred_rev)



# negative

test_text_rev =  "لم تعجبنى نظافة المكان والطعام سيء، لن أعود إلى المكان مستقبلا. نجمة واحدة." #  (category -1)

pred_rev = learn_c_rev.predict(test_text_rev)

print(pred_rev)
import matplotlib.pyplot as plt

import matplotlib.cm as cm

text_data ="لم تعجبنى نظافة المكان والطعام سيء، لن أعود إلى المكان مستقبلا. نجمة واحدة."

prediction = learn_c_rev.predict(text_data)

txt_ci = TextClassificationInterpretation.from_learner(learn_c_rev)

txt_ci.show_intrinsic_attention(text_data,cmap=plt.cm.Purples)

# The darker the word-shading in the below example, the more it contributes to the classification. 

# if the output is not highlighted in shades of purple, run the kernel interactively to see the interpretation.
test_data ="موقع الفندق جميل لكن وجبة الفطور سيئة جدا"

prediction = learn_c_rev.predict(test_data)

txt_ci.show_intrinsic_attention(test_data,cmap=plt.cm.Purples)
prediction
learn_c_rev.export('/kaggle/working/reviews_classifier_export.pkl')

!ls -la
learn_exp = load_learner('/kaggle/working/','reviews_classifier_export.pkl')
test_data ="موقع الفندق جميل لكن وجبة الفطور سيئة جدا لا انصح به ابدا"

prediction = learn_c_rev.predict(test_data)

prediction
# if you have a list of test items in a dataframe .............
# (first 5) https://www.tripadvisor.com/Restaurant_Review-g295424-d1584596-Reviews-Amaseena-Dubai_Emirate_of_Dubai.html#REVIEWS

# (last one) https://www.tripadvisor.com/Restaurant_Review-g295424-d940601-Reviews-or10-Arabian_Tea_House_Restaurant_Cafe_Al_Fahidi-Dubai_Emirate_of_Dubai.html

list_revews = [

    "أنصحكم بزياره البوفيه اكل متنوع وطعم ممتاز الخدمه مميزه خاصه من جاكسون مايكل وانا زيارتي له دائما تحياتي",

    "ابتسامه خدمة رائعه اكل طبب جدا مكان جميل رااااااااااااائع فب فندق الربتز خدمات رافية Jackson خدمته رائعه",

    "الاكل جميل ولذيذ ومتنوع ولكن الأسعار مبالغ فيها والعصاير خارج سعر البوفيه.... اشكرالموظف كريستيان على الخدمه",

    "مكان قمۃ في الفخامۃ والخدمۃ خمس نجوم ومن ارقي انواع الاستقبال ...ارشح زيارۃ هذا المكان والاستمتاع بجو من الملكيۃ...وطاقم مميز من العاملين حسام وجاكسون",

    "واحد من أجمل المطاعم بالعالم وخاصة عندما تتعرف على الموظفين وطريقة تعاملهم بفنون الأكل وخصوصا الموظف حسام الحريري الذي كان رائع جدا وانا احترمه كثيرا لانه صاحب مهارة عالية وخبير جدا في عمله واتمنى له التوفيق مما يدل على قدرة هذا المطعم في اخيار الموظفين الجيدين والاكفاء", 

    "يا جماعة الخير احنا جاين من بو ظبى لدبى فى اجتماع... الاكل ممتاز لكن لما اطلب شيئ وينزل شيئ ثانى امر يعصب عندنا موعد مهم وهذا اللى اسمو مسؤول احمد بالمساء اتاخرنا بسببه بعد ما نزل الطلب بالغلط راح يفزعنا كمان كل شيئ تمام . لا طبعا بهذا الاسلوب مو بتمام عمى . روح اتعلم "

]

import pandas as pd

test_df = pd.DataFrame(list_revews, columns=['text'])

test_df.head()
#test_df = pd.read_csv(path_to_test_csv_file)

learn_c_rev.data.add_test(test_df['text'])

prob_preds = learn_c_rev.get_preds(ds_type=DatasetType.Test, ordered=True)

prob_preds # first 4 examples above were positive, last one was given 1 star (negative) => wrong classification if 50% cut-off
preds, _ = learn_c_rev.get_preds(ds_type=DatasetType.Test, ordered=True)

thresh = 0.5

labelled_preds = [' '.join([str(learn_c_rev.data.classes[i]) for i,p in enumerate(pred) if p > thresh]) for pred in preds]

labelled_preds
#=========================== reviews, 3 lasses: +ve, -ve, mixed ===================

@np_func

def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1), average='weighted')
learn_3c = load_learner('/kaggle/input/arabicmf/','ar_classifier_reviews_sp15_multifit_nows_2fp_exp.pkl')
classes = ['Mixed','Negative', 'Positive']

#text_data ="لم تعجبنى نظافة المكان والطعام سيء، لن أعود إلى المكان مستقبلا. نجمة واحدة."

text_data  ="وجبة الفطور سيئة جدا لكن موقع الفندق جميل لا انصح به ابدا "

prediction = learn_3c.predict(text_data)

idx_class = prediction[1].item()

probs = [{ 'class': classes[i], 'probability': round(prediction[2][i].item(), 5) } for i in range(len(prediction[2]))]

result = {

        'idx_class': idx_class,

        'class name': classes[idx_class],

        'probability': round(prediction[2][idx_class].item(), 5),

        'list_prob': probs

    }



print(result)

print(probs)