def wscore(letters):

    if letters<4:

        score=0

    elif letters==4:

        score=1

    else:

        score=letters

    return(score)

main_file_path = '../input/enable1.txt' 

wordlist=open(main_file_path,'r').read().split('\n')

wordlist=[x for x in wordlist if (len(x)>3 and (not 's' in x))]

wordlengths=[len(x) for x in wordlist]

wordscores=[wscore(x) for x in wordlengths]

setlist=[{x for x in y} for y in wordlist]
alphabet='abcdefghijklmnopqrstuvwxyz'

for letter in alphabet:

    print(f"{letter} {len([x for x in setlist if letter in x])}")

def scoreletterset(string7,wordlist,dumpresults=False):

    centerletter=string7[0]

    sbset=set(string7)

    validwordlist=[x for x in wordlist if (centerletter in x and set(x).issubset(sbset))]

    validwordscores=[wscore(len(x)) for x in validwordlist]

    #check for pangrams

    pangramlist=[x for x in validwordlist if (sbset.issubset(set(x)))]

    regular_score=sum(validwordscores)

    bonus_score=len(pangramlist)*7

    total_score=regular_score+bonus_score

    print(f"{centerletter} {string7[1:]} {len(validwordlist)} words and {len(pangramlist)} pangrams for a total score of {total_score}")

    if dumpresults:

        print("WORDS")

        print(validwordlist)

        print("------------PANGRAMS--------------")

        print(pangramlist)

        print(f"average word score {sum(validwordscores)/len(validwordscores)}")

        print(f"maximum word length {max(validwordscores)}")
scoreletterset('eiarnto',wordlist,True)
def rotate7(rotate,wordlist):

    for letter in rotate:

        set7=set(rotate)

        set6=set7.copy()

        set6.remove(letter)

        string7=''

        string7+=letter

        for l in set6:string7+=l

        scoreletterset(string7,wordlist)
rotate7('eiarnto',wordlist)
rotate7('etaoinh',wordlist)
wordset_dict={w:frozenset(w) for w in wordlist}

possible_pangrams={k:v for k,v in wordset_dict.items() if len(v)==7}

len(possible_pangrams)
pangram_sets=list(possible_pangrams.values())

d={}

for s in pangram_sets:

    d[s] = d.get(s,0)+1

pangram_counts=list(d.values())

sorted(pangram_counts, reverse=True)

for key in d:

    if d[key]>=25:

        print(f"{key} {d[key]}")

        rotate7(''.join(list(key)),wordlist)

        #print(d[key])

        
scoreletterset('rigante',wordlist,True)