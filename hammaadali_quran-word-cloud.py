!pip install arabic_reshaper
!pip install python-bidi
!pip install requests
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import arabic_reshaper
import arabic_reshaper
from bidi.algorithm import get_display
import requests as r
from bs4 import BeautifulSoup
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/the-holy-quran/en.yusufali.csv')

eng_corpus = []
for num in range(1,115):
    full_surah = ""
    for aya in df[df.Surah==num].Text:
        full_surah +=  aya + " "
    eng_corpus.append(full_surah)
path = '../input/quran-clean-without-araab/Quran-clean-without-aarab.csv'
dff = pd.read_csv(path, index_col=None)

arabic_corpus = []
for num in range(1,115):
    full_surah = ""
    for aya in dff[dff.SurahNum==num].Ayah:
        full_surah +=  aya + " "
    arabic_corpus.append(full_surah)
#fetched from https://en.wikipedia.org/wiki/Meccan_surah
mec_surah_list = [96, 68, 73, 74, 1, 111, 81, 87, 92, 89,
                    93, 94, 103, 100, 108, 102, 107, 109, 105, 113,
                    114, 112, 53, 80, 97, 91, 85, 95, 106, 101,
                    75, 104, 77, 50, 90, 86, 54, 38, 7, 72,
                    36, 25, 35, 19, 20, 56, 26, 27, 28, 17,
                    10, 11, 12, 15, 6, 37, 31, 34, 39, 40,
                    41, 42, 43, 44, 45, 46, 51, 88, 18, 16,
                    71, 14, 21, 23, 32, 52, 67, 69, 70, 78,
                    79, 82, 84, 30, 29, 83]

eng_meccan_corp, eng_medinan_corp = "", ""
arabic_meccan_corp, arabic_medinan_corp = "", ""

for i in range(114):
        if i+1 in mec_surah_list:
            eng_meccan_corp += eng_corpus[i]
            arabic_meccan_corp += arabic_corpus[i]
        else:
            eng_medinan_corp += eng_corpus[i]
            arabic_medinan_corp += arabic_corpus[i]
# ### to obtain redundant words based on their frequency from the quran itself in ENGLISH
# #### ITS OUTPUT WILL BE USED IN THE STOPWORDS CELLL

# def get_uns(corpus):
#     uniques = {}
#     for s in corpus:
#         for w in s.split(' '):
#             if w not in uniques.keys():
#                 uniques[w] = 1
#             else:
#                 uniques[w] += 1
#     uniques = {k: v for k, v in sorted(uniques.items(), key=lambda item: item[1], reverse=True)}
#     return uniques

# get_uns(corpus)
# get_uns(arabic_corpus)
#### all the surahs names fetched from iternet

surah_name = """1. Al-Fatihah (the Opening)
2. Al-Baqarah (the Cow)
3. Aali Imran (the Family of Imran)
4. An-Nisa’ (the Women)
5. Al-Ma’idah (the Table)
6. Al-An’am (the Cattle)
7. Al-A’raf (the Heights)
8. Al-Anfal (the Spoils of War)
9. At-Taubah (the Repentance)
10. Yunus (Yunus)
11. Hud (Hud)
12. Yusuf (Yusuf)
13. Ar-Ra’d (the Thunder)
14. Ibrahim (Ibrahim)
15. Al-Hijr (the Rocky Tract)
16. An-Nahl (the Bees)
17. Al-Isra’ (the Night Journey)
18. Al-Kahf (the Cave)
19. Maryam (Maryam)
20. Ta-Ha (Ta-Ha)
21. Al-Anbiya’ (the Prophets)
22. Al-Haj (the Pilgrimage)
23. Al-Mu’minun (the Believers)
24. An-Nur (the Light)
25. Al-Furqan (the Criterion)
26. Ash-Shu’ara’ (the Poets)
27. An-Naml (the Ants)
28. Al-Qasas (the Stories)
29. Al-Ankabut (the Spider)
30. Ar-Rum (the Romans)
31. Luqman (Luqman)
32. As-Sajdah (the Prostration)
33. Al-Ahzab (the Combined Forces)
34. Saba’ (the Sabeans)
35. Al-Fatir (the Originator)
36. Ya-Sin (Ya-Sin)
37. As-Saffah (Those Ranges in Ranks)
38. Sad (Sad)
39. Az-Zumar (the Groups)
40. Ghafar (the Forgiver)
41. Fusilat (Distinguished)
42. Ash-Shura (the Consultation)
43. Az-Zukhruf (the Gold)
44. Ad-Dukhan (the Smoke)
45. Al-Jathiyah (the Kneeling)
46. Al-Ahqaf (the Valley)
47. Muhammad (Muhammad)
48. Al-Fat’h (the Victory)
49. Al-Hujurat (the Dwellings)
50. Qaf (Qaf)
51. Adz-Dzariyah (the Scatterers)
52. At-Tur (the Mount)
53. An-Najm (the Star)
54. Al-Qamar (the Moon)
55. Ar-Rahman (the Most Gracious)
56. Al-Waqi’ah (the Event)
57. Al-Hadid (the Iron)
58. Al-Mujadilah (the Reasoning)
59. Al-Hashr (the Gathering)
60. Al-Mumtahanah (the Tested)
61. As-Saf (the Row)
62. Al-Jum’ah (Friday)
63. Al-Munafiqun (the Hypocrites)
64. At-Taghabun (the Loss & Gain)
65. At-Talaq (the Divorce)
66. At-Tahrim (the Prohibition)
67. Al-Mulk – (the Kingdom)
68. Al-Qalam (the Pen)
69. Al-Haqqah (the Inevitable)
70. Al-Ma’arij (the Elevated Passages)
71. Nuh (Nuh)
72. Al-Jinn (the Jinn)
73. Al-Muzammil (the Wrapped)
74. Al-Mudaththir (the Cloaked)
75. Al-Qiyamah (the Resurrection)
76. Al-Insan (the Human)
77. Al-Mursalat (Those Sent Forth)
78. An-Naba’ (the Great News)
79. An-Nazi’at (Those Who Pull Out)
80. ‘Abasa (He Frowned)
81. At-Takwir (the Overthrowing)
82. Al-Infitar (the Cleaving)
83. Al-Mutaffifin (Those Who Deal in Fraud)
84. Al-Inshiqaq (the Splitting Asunder)
85. Al-Buruj (the Stars)
86. At-Tariq (the Nightcomer)
87. Al-A’la (the Most High)
88. Al-Ghashiyah (the Overwhelming)
89. Al-Fajr (the Dawn)
90. Al-Balad (the City)
91. Ash-Shams (the Sun)
92. Al-Layl (the Night)
93. Adh-Dhuha (the Forenoon)
94. Al-Inshirah (the Opening Forth)
95. At-Tin (the Fig)
96. Al-‘Alaq (the Clot)
97. Al-Qadar (the Night of Decree)
98. Al-Bayinah (the Proof)
99. Az-Zalzalah (the Earthquake)
100. Al-‘Adiyah (the Runners)
101. Al-Qari’ah (the Striking Hour)
102. At-Takathur (the Piling Up)
103. Al-‘Asr (the Time)
104. Al-Humazah (the Slanderer)
105. Al-Fil (the Elephant)
106. Quraish (Quraish)
107. Al-Ma’un (the Assistance)
108. Al-Kauthar (the River of Abundance)
109. Al-Kafirun (the Disbelievers)
110. An-Nasr (the Help)
111. Al-Masad (the Palm Fiber)
112. Al-Ikhlas (the Sincerity)
113. Al-Falaq (the Daybreak)
114. An-Nas (Mankind)"""

surah_name = list(map(lambda x: x.split(' ')[1], (surah_name.split('\n'))))


site =  r.get('https://gpsarab.com/shop11/en/content/11-list-of-surahs-in-the-holy-quran').content
soup = BeautifulSoup(site, "html.parser")
surah_name_arabic = [s.text for s in soup.findAll('td')]
surah_name_arabic = [surah_name_arabic[x] for x in range(1, len(surah_name_arabic),5)]

for (i, x ),y  in zip(enumerate(surah_name),surah_name_arabic):
    print("{} -> {} -> {}".format(i,x,y))
##### STOPWORDS CELL  ######


### to get redundant words(pronouns etc) so we can omit them later on
SW = list(STOPWORDS) + ['ye', 'verily', 'will', 'said', 'say', 'us', 'thy', 'thee', 'thou',
                        'the', 'and', 'of', 'to', 'is', 'in', 'they', 'a', 'that', 'for', 
                        'ye', 'who', 'their', 'not', 'them', 'He', 'be', 
                        'We', 'those', 'with', 'have', 'are', 'And', 'from', 'it', 'but', 
                        'on', 'you', 'your', 'all', 'as', 'he', 'shall', 'if', 'thou', 'no',
                        'which', 'But', 'do', 'his', 'what', 'I', 'or', 'when', 'we', 'by', 
                        'His', 'said:', 'thy', 'has', 'this', 'They', 'there', 
                        'then', 'one', 'my', 'him', 'were', 'was', 'thee', 'them,', 'may', 'any',
                        'had', 'sent', 'before', 'nor', 'among', 'whom', 'Day', 'hath', 'made', 
                        'did', '(of', 'Who', 'would', '(in', 'out', 'Say:', 'our', 'indeed',
                        'so', 'If', '(to', '(the', 'against', 'been', 'an', 'For', 'you,', 
                        'us', 'The', 'Then', 'fear', 'than', 'give', '-', 'should', 'such', 'Most',
                        'down', 'men', 'So', 'say:', '"O', 'Our', 'It', 'come', 'can', 'after', 'O', 
                        'me', 'some', 'turn', '', 'over', 'up', 'things', 'make', 'know',
                        'reject', 'When', 'unto', 'into', 'its', 'see', 'Those', 'only', 
                        'them:','good', 'own', 'doth', 'of)', 'most', 'other', 
                        'except', '(for', 'Thou', 'at', '(and', 'between', 'take', 'away',
                        'given', 'every', 'back', 'say,', 'verily', 'never', 'That', 'said'
                       'whose', 'where', 'which', 'how', 'when']

### arabic redundant words
s = ['من', 'في', 'ما',
       'إن', 'لا', 'على', 'إلا', 'ولا', 'وما', 'أن', 'قال', 'إلى', 'لهم', 'يا', 'ومن', 'ثم', 'لكم', 'به', 'كان', 'بما'
       , 'قل', 'ذلك', 'أو', 'له', 'الذي', 'هو',  'هم', 'وإن', 'قالوا', 'كل', 'فيها', 'كانوا', 'عن', 'إذا',  'عليهم', 
       'شيء', 'هذا', 'كنتم',  'لم', 'وهو', 'فإن', 'إذ',  'عليكم',  'إنا', 'فلا', 'منهم',  'أيها', 'إنه','بعد', 'عليه',
       'حتى', 'وهم', 'وإذا', 'أولئك', 'أم', 'إني', 'ولقد', 'فيه', 'بل', 'قد', 'عند', 'إنما', 'ولكن', 'ولو',
       'مما',  'منكم', 'فلما', 'ألا', 'لمن',  'دون', 'فمن', 'منه', 'فإذا', 'فما', 'منها', 'كذلك', 'وقال', 'وكان']
ASW1 = [get_display(arabic_reshaper.reshape(x)) for x in s]


### fetched this list of stopwords of arabic 
# from https://github.com/mohataher/arabic-stop-words/blob/master/list.txt
s = """ء
ءَ
آ
آب
آذار
آض
آل
آمينَ
آناء
آنفا
آه
آهاً
آهٍ
آهِ
أ
أبدا
أبريل
أبو
أبٌ
أجل
أجمع
أحد
أخبر
أخذ
أخو
أخٌ
أربع
أربعاء
أربعة
أربعمئة
أربعمائة
أرى
أسكن
أصبح
أصلا
أضحى
أطعم
أعطى
أعلم
أغسطس
أفريل
أفعل به
أفٍّ
أقبل
أكتوبر
أل
ألا
ألف
ألفى
أم
أما
أمام
أمامك
أمامكَ
أمد
أمس
أمسى
أمّا
أن
أنا
أنبأ
أنت
أنتم
أنتما
أنتن
أنتِ
أنه
أنًّ
أنّى
أهلا
أو
أوت
أوشك
أول
أولئك
أولاء
أولالك
أوّهْ
أى
أي
أيا
أيار
أيضا
أيلول
أين
أيّ
أيّان
أُفٍّ
ؤ
إذ
إذا
إذاً
إذما
إذن
إزاء
إلى
إلي
إليكم
إليكما
إليكنّ
إليكَ
إلَيْكَ
إلّا
إمّا
إن
إنَّ
إى
إياكم
إياكما
إياكن
إيانا
إياه
إياها
إياهم
إياهما
إياهن
إياي
إيهٍ
ئ
ا
ا?
ا?ى
االا
االتى
ابتدأ
ابين
اتخذ
اثر
اثنا
اثنان
اثني
اثنين
اجل
احد
اخرى
اخلولق
اذا
اربعة
اربعون
اربعين
ارتدّ
استحال
اصبح
اضحى
اطار
اعادة
اعلنت
اف
اكثر
اكد
الآن
الألاء
الألى
الا
الان
الاولى
التى
التي
الحالي
الذاتي
الذى
الذي
الذين
السابق
الف
اللاتي
اللتان
اللتيا
اللتين
اللذان
اللذين
اللواتي
الماضي
المقبل
الوقت
الى
الي
اليه
اليها
اما
امام
امس
امسى
ان
انبرى
انقلب
انه
انها
او
اول
اي
ايار
ايام
ايضا
ب
بؤسا
بإن
بئس
باء
بات
باسم
بان
بخٍ
بد
بدلا
برس
بسبب
بسّ
بشكل
بضع
بطآن
بعد
بعدا
بعض
بغتة
بل
بلى
بن
به
بها
بهذا
بيد
بين
بَسْ
بَلْهَ
ة
ت
تاء
تارة
تاسع
تانِ
تانِك
تبدّل
تجاه
تحوّل
تخذ
ترك
تسع
تسعة
تسعمئة
تسعمائة
تسعون
تسعين
تشرين
تعسا
تعلَّم
تفعلان
تفعلون
تفعلين
تكون
تلقاء
تلك
تم
تموز
تينك
تَيْنِ
تِه
تِي
ث
ثاء
ثالث
ثامن
ثان
ثاني
ثلاث
ثلاثاء
ثلاثة
ثلاثمئة
ثلاثمائة
ثلاثون
ثلاثين
ثم
ثمان
ثمانمئة
ثمانون
ثماني
ثمانية
ثمانين
ثمنمئة
ثمَّ
ثمّ
ثمّة
ج
جانفي
جدا
جعل
جلل
جمعة
جميع
جنيه
جوان
جويلية
جير
جيم
ح
حاء
حادي
حار
حاشا
حاليا
حاي
حبذا
حبيب
حتى
حجا
حدَث
حرى
حزيران
حمدا
حمو
حمٌ
حوالى
حول
حيث
حيثما
حين
حيَّ
حَذارِ
خ
خاء
خاصة
خال
خامس
خلا
خلافا
خلال
خلف
خمسة
خمسمئة
خمسمائة
خمسون
خمسين
خميس
د
دال
درهم
درى
دواليك
دولار
دون
دونك
ديسمبر
دينار
ذ
ذا
ذات
ذاك
ذال
ذانك
ذانِ
ذلك
ذهب
ذو
ذيت
ذينك
ذَيْنِ
ذِه
ذِي
ر
رأى
راء
رابع
راح
رجع
رزق
رويدك
ريال
ريث
ز
زاي
زعم
زود
زيارة
س
ساء
سادس
سبت
سبتمبر
سبحان
سبعة
سبعمئة
سبعمائة
سبعون
سبعين
ست
ستة
ستكون
ستمئة
ستمائة
ستون
ستين
سحقا
سرا
سرعان
سقى
سمعا
سنتيم
سنوات
سوف
سوى
سين
ش
شباط
شبه
شتانَ
شخصا
شرع
شيكل
شين
شَتَّانَ
ص
صاد
صار
صباح
صبر
صبرا
صدقا
صراحة
صفر
صهٍ
صهْ
ض
ضاد
ضحوة
ضد
ضمن
ط
طاء
طاق
طالما
طرا
طفق
طَق
ظ
ظاء
ظل
ظلّ
ظنَّ
ع
عاد
عاشر
عام
عاما
عامة
عجبا
عدا
عدة
عدد
عدم
عدَّ
عسى
عشر
عشرة
عشرون
عشرين
عل
علق
علم
على
علي
عليك
عليه
عليها
علًّ
عن
عند
عندما
عنه
عنها
عوض
عيانا
عين
عَدَسْ
غ
غادر
غالبا
غدا
غداة
غير
غين
ـ
ف
فإن
فاء
فان
فانه
فبراير
فرادى
فضلا
فقد
فقط
فكان
فلان
فلس
فهو
فو
فوق
فى
في
فيفري
فيه
فيها
ق
قاطبة
قاف
قال
قام
قبل
قد
قرش
قطّ
قلما
قوة
ك
كأن
كأنّ
كأيّ
كأيّن
كاد
كاف
كان
كانت
كانون
كثيرا
كذا
كذلك
كرب
كسا
كل
كلتا
كلم
كلَّا
كلّما
كم
كما
كن
كى
كيت
كيف
كيفما
كِخ
ل
لأن
لا
لا سيما
لات
لازال
لاسيما
لام
لايزال
لبيك
لدن
لدى
لدي
لذلك
لعل
لعلَّ
لعمر
لقاء
لكن
لكنه
لكنَّ
للامم
لم
لما
لمّا
لن
له
لها
لهذا
لهم
لو
لوكالة
لولا
لوما
ليت
ليرة
ليس
ليسب
م
مئة
مئتان
ما
ما أفعله
ما انفك
ما برح
مائة
ماانفك
مابرح
مادام
ماذا
مارس
مازال
مافتئ
ماي
مايزال
مايو
متى
مثل
مذ
مرّة
مساء
مع
معاذ
معه
مقابل
مكانكم
مكانكما
مكانكنّ
مكانَك
مليار
مليم
مليون
مما
من
منذ
منه
منها
مه
مهما
ميم
ن
نا
نبَّا
نحن
نحو
نعم
نفس
نفسه
نهاية
نوفمبر
نون
نيسان
نيف
نَخْ
نَّ
ه
هؤلاء
ها
هاء
هاكَ
هبّ
هذا
هذه
هل
هللة
هلم
هلّا
هم
هما
همزة
هن
هنا
هناك
هنالك
هو
هي
هيا
هيهات
هيّا
هَؤلاء
هَاتانِ
هَاتَيْنِ
هَاتِه
هَاتِي
هَجْ
هَذا
هَذانِ
هَذَيْنِ
هَذِه
هَذِي
هَيْهات
و
و6
وأبو
وأن
وا
واضاف
واضافت
واكد
والتي
والذي
وان
واهاً
واو
واوضح
وبين
وثي
وجد
وراءَك
ورد
وعلى
وفي
وقال
وقالت
وقد
وقف
وكان
وكانت
ولا
ولايزال
ولكن
ولم
وله
وليس
ومع
ومن
وهب
وهذا
وهو
وهي
وَيْ
وُشْكَانَ
ى
ي
ياء
يفعلان
يفعلون
يكون
يلي
يمكن
ين
يناير
يوان
يورو
يوليو
يونيو
ّأيّان
""".split('\n')
ASW2 = [get_display(arabic_reshaper.reshape(x)) for x in s]


#### Obtained new combined corpus of redundant words
ASW = ASW1 + ASW2
####  This is the main function which will be used to generate various kinds of wordclouds

def generateWordCloud(surah=None, corpus=None, title=None, isArabic=False):
    plt.figure(figsize=(15,8))
    
    if not isArabic:
        wc = WordCloud(stopwords=SW, max_words = 80, width=800, height=400,
                       background_color='black')
        if surah is None:
            wc.generate(corpus)
            plt.title(title, fontdict={'fontsize':40})
        else:
            wc.generate(eng_corpus[surah])
            plt.title(surah_name[surah], fontdict={'fontsize':40})

    if isArabic:
        wc = WordCloud(font_path='../input/arial-font/arial.ttf', #relative_scaling=1,
                       stopwords=ASW, max_words=80, width=800, height=400,
                       background_color='black')
        if surah is None:
            wc = wc.generate_from_text(get_display(arabic_reshaper.reshape(corpus))) 
            plt.title(get_display(arabic_reshaper.reshape(title)), fontdict={'fontsize':40})
        else:
            wc = wc.generate_from_text(get_display(arabic_reshaper.reshape(arabic_corpus[surah])))
            plt.title(get_display(arabic_reshaper.reshape(surah_name_arabic[surah])), fontdict={'fontsize':40})

    plt.imshow(wc)
    plt.axis('off')
    
all_eng = ' '.join(eng_corpus)
generateWordCloud(corpus=all_eng, title='Entire Quran')
all_arabic = ' '.join(arabic_corpus)
generateWordCloud(corpus=all_arabic, title='Entire Quran Arabic', isArabic=True)
generateWordCloud(11, eng_corpus)
generateWordCloud(68, eng_corpus)
for i in range(len(eng_corpus)):
    generateWordCloud(i, eng_corpus)
generateWordCloud(54, isArabic=True, corpus=arabic_corpus)
generateWordCloud(55, isArabic=True, corpus=arabic_corpus)
for i in range(len(arabic_corpus)):
    generateWordCloud(i, arabic_corpus, isArabic=True)
generateWordCloud(corpus=eng_meccan_corp, title="Meccan(Makki) Surahs English")
generateWordCloud(corpus=eng_medinan_corp, title="Medinan(Madni) Surahs English")
generateWordCloud(corpus=arabic_meccan_corp, title="Meccan(Makki) Surahs Arabic", isArabic=True)
generateWordCloud(corpus=arabic_medinan_corp, title="Medinan(Madni) Surahs Arabic", isArabic=True)