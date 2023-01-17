def isNumber(num):
    if num == '.' or num == ',':
        return False
    if len(num) == 0 :
        return True
    nums = ['due', 'tre', 'quattro', 'cinque', 'sei', 'sette', 'otto', 'nove', 'dieci',
        'undici', 'dodici', 'tredici', 'quattordici', 'quindici', 'sedici', 'diciassette', 'diciotto', 'diciannove', 'venti',
        'ventuno', 'ventotto', 'trenta',
        'trentuno', 'trentotto', 'quaranta',
        'quarantuno', 'quarantotto', 'cinquanta',
        'cinquantuno', 'cinquantotto', 'sessanta',
        'sessantuno', 'sessantotto', 'settanta',
        'settantuno', 'settantotto', 'ottanta',
        'ottantuno', 'ottantotto', 'novanta',
        'novantuno', 'novantotto', 'cento',
        'mille', 'milione', 'milioni', 'mila', 'miliardo', 'miliardi']

    b = True
    for i in range(len(num)):
        y = ord(num[i])
        if not ((y < 58 and y > 47) or y == 46 or y == 44) :
            b = False
            break
    if b: 
        return True

    for el in nums:
        if num.startswith(el):
            return isNumber(num[len(el):len(num)])
    return False
class prepara_dati():
    def dividi(self, text):
        words = []
        tags = []
        dizionario = {}
        tot = {'ADJ':0,'NOUN':0,'ADP':0,'DET':0,'PROPN':0,'PUNCT':0,'AUX':0,'VERB':0,'PRON':0,'CCONJ':0,'NUM':0,'ADV':0,'INTJ':0,'SCONJ':0,'X':0,'SYM':0,'PART':0, 'tot':0}
        for frase in text:
            words.append([])
            tags.append([])
            for parola in frase.split(" "):
                spl = parola.split("_")
                if len(spl) != 1 and len(spl[1]) != 0:
                    temp = (spl[0]).strip().rstrip("\n").lower()
                    p =  temp if not isNumber(temp) else "&spnum&" 
                    dizionario[p] = 1 if p not in dizionario else dizionario[p] + 1
                    spl[1] = spl[1].strip().rstrip("\n")
                    words[-1].append(p)
                    tags[-1].append(spl[1])
                    tot['tot'] += 1
                    tot[spl[1]] += 1
        return words, tags, dizionario, tot
class hmm:        
    # inizializza i dati statici necessari 
    def inizializza(self):
        self.useless = ["DET", "ADP", 'PUNCT', 'PRON', "SCONJ", "CCONJ", "NUM", "INTJ", "X", "SYM", "PART"]
        self.states = ['ADJ','NOUN','ADP','DET','PROPN','PUNCT','AUX','VERB','PRON','CCONJ','NUM','ADV','INTJ','SCONJ','X','SYM','PART']
    
    # inizializza i dati dinamici e chiama le funzioni per addestrare il modello
    def addestra(self, train):
        self.inizializza()
        self.prepara_dati = prepara_dati()
        self.train_word, self.train_tags, self.dict, self.tot = self.prepara_dati.dividi(train) 
        self.addestra_rare()
        self.crea_transition()
        self.crea_emission()
    
    # sostituisce all'interno del train le parole che appaiono meno di 2 volte con il token &rare& e le cancella dal dizionario
    def addestra_rare(self):
        keys = []
        for frase in range(len(self.train_word)):
            for parola in range(len(self.train_word[frase])):
                act = self.train_word[frase][parola]
                if self.dict[act] <= 1:
                    if act not in keys:
                        keys.append(act)
                    self.train_word[frase][parola] = "&rare&"
        for k in keys:
            del self.dict[k]
    
    # crea la matrice transition
    def crea_transition(self): 
        transition = {'IN':{},'OUT':{}}
        for frase in range(len(self.train_tags)):
            for parola in range(len(self.train_tags[frase])):
                tag = self.train_tags[frase][parola]
                if parola == 0:
                    transition['IN'][tag] = 1 if tag not in transition['IN'] else transition['IN'][tag] + 1                        
                if parola+1 == len(self.train_tags[frase]): 
                    transition['OUT'][tag] = 1 if tag not in transition['OUT'] else transition['OUT'][tag] + 1
                else:
                    tagPost = self.train_tags[frase][parola+1]
                    transition[tag] = {} if tag not in transition else transition[tag]
                    transition[tag][tagPost] = 1 if tagPost not in transition[tag] else transition[tag][tagPost]+1           
        for k in transition:
            for kk in transition[k]:
                transition[k][kk] = transition[k][kk]/self.tot['tot'] if k == "IN" or k == "OUT" else transition[k][kk]/self.tot[k]
        self.transition = transition
    
    #crea la matrice emission
    def crea_emission(self):
        t = {}
        emission = {'ADJ':{},'NOUN':{},'ADP':{},'DET':{},'PROPN':{},'PUNCT':{},'AUX':{},'VERB':{},'PRON':{},'CCONJ':{},'NUM':{},'ADV':{},'INTJ':{},'SCONJ':{},'X':{},'SYM':{},'PART':{}};
        for frase in range(len(self.train_tags)):
            for parola in range(len(self.train_tags[frase])):
                w = self.train_word[frase][parola]
                tag = self.train_tags[frase][parola]
                emission[tag] = {} if not tag in emission else emission[tag]
                emission[tag][w] = 1 if w not in emission[tag] else emission[tag][w]+1
                t[w] = 1 if w not in t else t[w]+1
        for k in emission:
            for kk in emission[k]:
                emission[k][kk] = emission[k][kk]/t[kk]
        self.emission = emission
  
    # controlla gli stati con cui può iniziare la sequenza
    def valuta_inizio(self, obs):
        # controllo che stati può prendere il primo elemento
        V = [{}]
        osservati = [stato  for stato in self.states  if stato in self.emission and obs[0] in self.emission[stato]]
        # calcolo le probabilità di inizio con un determinato stato
        for s in self.states:
            if len(osservati) > 0:
                V[0][s] = {'prob': self.transition["IN"][s]*self.emission[s][obs[0]], 'prev':0} if s in osservati and s in self.emission and s in self.transition["IN"] else {'prob':0,'prev':0}
            else:
                V[0][s] ={'prob': 10**(-8), 'prev':0 } if s not in self.useless else {'prob': 0, 'prev':0} 
        return V
    
    # calcola tutte le possibili strade con le relative probabilità
    def valuta(self, obs):
        V = self.valuta_inizio(obs)
        # calcolo le strade più probabili
        for t in range(1, len(obs)):
            act = obs[t].lower().strip()
            V.append({})
            # calcolo tutte le strade e tengo la più probabile
            for state in self.states:
                max_tr_prob = V[t-1][self.states[0]]['prob'] * self.transition[self.states[0]][state] if state in self.transition[self.states[0]] else 0
                prev_st_selected = self.states[0]
                for prev_st in self.states:
                    if prev_st in self.transition:
                        tr_prob = V[t-1][prev_st]['prob'] * self.transition[prev_st][state] if state in self.transition[prev_st] else 0
                        if tr_prob > max_tr_prob:
                            max_tr_prob = tr_prob
                            prev_st_selected = prev_st
                # calcolo la probabilità di emissione
                em = self.emission[state][act] if state in self.emission and act in self.emission[state] else 10**(-8)
                max_prob = max_tr_prob * em
                V[t][state] = {'prob': max_prob, 'prev': prev_st_selected}
        return V
       
    #sostituisce ai numeri lo speciale token &spnum&
    def converti_numeri(self, obs):
        return ["&spnum&" if isNumber(obs[el].strip().lower()) else obs[el].strip().lower() for el in range(len(obs))]

    # sostituisce alle parole non nel dizionario il token &rare&
    def converti_rare(self, obs):
        return [obs[el] if obs[el] in self.dict else "&rare&" for el in range(len(obs))]
    
    # algoritmo di viterbi per il calcolo delle strade possibili
    def viterbi(self, obs):
        obs = self.converti_rare(self.converti_numeri(obs))
        V = self.valuta(obs)
        opt = []
        # The highest probability
        max_prob = max([V[-1][e]['prob'] for e in V[-1]])
        
        # trovo l'ultimo
        for i in V[-1]:
            if V[-1][i]['prob'] == max_prob:
                opt.append(i)
                previous = i
                break
        # ricostruisco la sequenza
        for t in reversed(range(len(V) - 1)):
            opt.insert(0,V[t + 1][previous]['prev'])
            previous = V[t + 1][previous]['prev']
        return opt
h = hmm()
h.addestra(open('/kaggle/input/train.txt', 'r', encoding='utf-8-sig'))
p = prepara_dati()
x_test, y_test, d, t = p.dividi(open('/kaggle/input/test.txt', 'r', encoding='utf-8-sig'))
len(y_test)
import time
def prova(x, y, h):
    ts_w = x
    ts_t = y
    tot = 0
    giuste = 0
    ug = ['NOUN', 'PROPN']
    in_t=time.time()
    for frase in range(len(ts_w)):
        f = [e.replace('\ufeff', '') for e in ts_w[frase]]
        pred = h.viterbi(f)
        for e in range(len(ts_t[frase])):
            t = ts_t[frase][e].strip() 
            if  t == pred[e] or (pred[e] in ug and t in ug):
                giuste += 1
            tot += 1
    print('totali : {} , corrette : {} = {} %'.format(tot, giuste, (giuste/tot)*100))
    out_t = time.time()-in_t
    print("tempo impiegato : {}".format(out_t))
prova(x_test, y_test, h)            

def trova(parola):
    for s in h.emission:
        if parola in h.emission[s]:
            print('stato : {} , parola : {}, prob : {} '.format(s,parola,h.emission[s][parola]))
trova('quando')
