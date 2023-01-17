class remove_puctuations():

    def __init__(self, filepaths):

        self.filepaths = filepaths

        self.punctuations = [".","'",'"',"(",")","-",",","/",";",":","?","%","!","|","[","]","{","}","_","@","#","$","`","~","<",">","�"]

        self.cleaned_corpus = []

    def remove(self):

        for i in range(len(self.filepaths)):

            with open(self.filepaths[i],"r", newline="", encoding="iso-8859-1") as docs:

                cleaned_data = []

                lines = docs.readlines()

                for line in lines:

                    for punc in self.punctuations:

                        line = line.replace(punc, "")

                    cleaned_data.append(line)

                self.cleaned_corpus.append(cleaned_data)

        return self.cleaned_corpus

    









class remove_stop_words:

    def __init__(self ,corpora):

        self.corpora = corpora

        self.checked_words = []

        self.cleaned_corpora = []

    def remove_words(self):

        for corpus in self.corpora:

            self.checked_words = []

            for line in corpus:

                self.check_stop_word(line.split(" "))

            processed_corpus = []

            size = len(self.checked_words)

            base = 0

            top = 10

            while True: #chuncking into words of tens

                if (size - top+1 > 10):

                    string  = " ".join(self.checked_words[base:top])

                    processed_corpus.append(string)

                    base = top

                    top += 11

                else:

                    string = " ".join(self.checked_words[base:top])

                    processed_corpus.append(string)

                    base = top

                    top = size

                    string = " ".join(self.checked_words[base:top])

                    processed_corpus.append(string)

                    self.cleaned_corpora.append(processed_corpus)

                    break

                #print(base, top)

        return self.cleaned_corpora



    def check_stop_word(self,line):

        with open("../input/stop_words.txt","r") as first_check:

            stopwords = first_check.readlines()

            stopword = []

            for stop in stopwords:

                stopword.append(stop[:-1])

            for word in line:

                word = word.lower()

                if "\r\n" in word:

                    word = word.replace("\r\n","")

                if word not in stopword:

                    self.checked_words.append(word)

                    

class stem_down:

    def __init__(self,corpora):

        from nltk.stem import PorterStemmer

        self.stm = PorterStemmer()

        self.corpora = corpora

        self.cleaned_corpora = []



    def stemm(self):

        for corpus in self.corpora:

            cleaned_corpus = []

            for sentence in corpus:

                word_li = []

                for word in sentence.split():

                    word_li.append(self.stm.stem(word))

                cleaned_corpus.append(word_li)

            self.cleaned_corpora.append(cleaned_corpus)

            

        return self.cleaned_corpora

                    

            



class cooccurence_matrix:

    def __init__(self,corpora):

        self.corpora = corpora

        self.word_union = []

        self.corpora_count = []

        self.corpus_count={}

    def union(self): #find the set of all words in corpora

        for corpus in self.corpora:

            for sentence in corpus:

                for word in sentence:

                    self.word_union.append(word)

        self.word_union = set(self.word_union)

        return self.word_union

        

    def coocurence_count(self):

        for corpus in self.corpora:

            for wod in self.word_union:

                cooccurence = 0

                for sentence in corpus:

                    for word in sentence:

                        if wod == word:

                            cooccurence +=1

                            

                if cooccurence != 0:

                    self.corpus_count[wod] = cooccurence

                    cooccurence = 0

            self.corpora_count.append(self.corpus_count)

            self.corpus_count = {}

        return self.corpora_count#its in order

    



class build_model:

    def __init__(self, word_vocab):

        self.vocabulary =  word_vocab

        self.total_words = 0

        self.corpus_tot = 0

        self.model = {}

        self.word_model = {}

    def pre_build(self):

        for diction in self.vocabulary:

            for key,val in diction.items():

                self.total_words+=int(val)

    def build(self, vocab_union,symptoms_list):

        symptom_counter = 0

        for diction in self.vocabulary:

            freq_prob = 0

            for key, val in diction.items():

                self.corpus_tot += int(val)

            for word in vocab_union:

                #print(diction)

                if word in diction.keys():

                    freq_prob = (int(diction[word])+1)/(self.corpus_tot + (self.total_words-int(diction[word])+1))

                    #print(freq_prob)

                else:

                    freq_prob = 1/(self.corpus_tot+(self.total_words-1))

                    #print(freq_prob)

                self.word_model[word] = freq_prob

            self.model[symptoms_list[symptom_counter]]=self.word_model

            symptom_counter =  symptom_counter + 1

            #print(self.word_model)

            #print("\n\n\n\n")

            self.corpus_tot = 0

            self.word_model = {}

        return self.model

            



    def write_to_file(self,symp_name):

        ok = {}

        for i in range(len(symp_name)):

            ok[symp_name[i]] = self.model[i]

        import pickle

        pickle.dump(ok, open("model.p","wb"))

        print("done")

        

        



        

        

        

        

        

        

class detemine_symptom:

    def __init__(self, sentence, model):

        self.sentence = sentence

        self.word_list = []

        self.model = model

        self.out_of_bounds = []

        self.symptoms_list = ["fever.n.01","chill.n.03","headache.n.02","vomit.v.01","Asthenia.n.01","dizziness.n.01",

                              "arthralgia.n.01","stomachache.n.01","diarrhea.n.01","thirst.v.01","spasm.n.01","restlessness.n.04",

                              "cough.n.01","dizziness.n.01","anorexia.n.01","sneeze.n.01","sore_throat.n.01"]

    def greater_val(self, dict):

        greatest = 0

        symp = ""

        for k,v in dict.items():

            if v > greatest:

                greatest = v

                symp = str(k)

        return symp

    def remove_stop_words(self):

        stop_words = []

        words = []

        with open('../input/stop_words.txt',"r")as doc:

                stop_words = doc.readlines()

        for word in self.sentence.split(" "):

            if word+"\r\n" not in stop_words:

                words.append(word)

        return " ".join(words)





    def determ(self):

        from nltk.stem import PorterStemmer

        stm = PorterStemmer()

        self.sentence = self.remove_stop_words()

        for word in self.sentence.split(" "):

            self.word_list.append(stm.stem(word))

        class_prob = {}

        for key,val in self.model.items():

            prob = 1

            for word in self.word_list:

                if word in val.keys():

                    #print(key,word,val[word])

                    prob*=val[word]

            class_prob[key] = prob

        print(self.greater_val(class_prob))

        return self.greater_val(class_prob)







#model build process###

stage1 = remove_puctuations(["../input/fever.txt","../input/arthralgia.txt","../input/asthenia.txt","../input/chill.txt","../input/diarrhea.txt","../input/headache.txt","../input/Perspiration.txt","../input/restlessness.txt","../input/stomachache.txt","../input/thirst.txt","../input/Vomiting.txt","../input/Spasm.txt","../input/Bitter_mouth.txt",

                    "../input/Cough.txt","../input/Dizziness.txt","../input/Inapetence.txt","../input/Sneezing.txt","../input/Sore_Throat.txt",])

cleaned_data1 = stage1.remove()



stage2 = remove_stop_words(cleaned_data1)

cleaned_data2 = stage2.remove_words()



stage3 = stem_down(cleaned_data2)

cleaned_data3 = stage3.stemm()



stage4 = cooccurence_matrix(cleaned_data3)

vocabulary = stage4.union()

cleaned_data4 = stage4.coocurence_count()



stage5 = build_model(cleaned_data4)



model = stage5.build(vocabulary,["fever","arthralgia","asthenia","chill","diarrhea","headache","Perspiration","restlessness","stomachache","thirst","Vomiting",

"Spasm","Bitter_mouth","Cough","Dizziness","Inapetence","Sneezing","Sore_Throat"])





#Test

prediction = detemine_symptom("I have a headache" ,model)



prediction.determ()

                            

                