def phrase_extraction(srctext, trgtext, alignment, first=1):

    """

    Phrase extraction algorithm.

    """

    # Convert to a 0-based indexing.

    if first != 0:

        alignment = [(x-first, y-first) for x, y in alignment]



    def extract(f_start, f_end, e_start, e_end):

        if f_end < 0:  # 0-based indexing.

            return {}

        # Check if alignement points are consistent.

        for e,f in alignment:

            if ((f_start <= f <= f_end) and

               (e < e_start or e > e_end)):

                return {}



        # Add phrase pairs (incl. additional unaligned f)

        # Remark:  how to interpret "additional unaligned f"?

        phrases = set()

        fs = f_start

        # repeat-

        while True:

            fe = f_end

            # repeat-

            while True:

                # add phrase pair ([e_start, e_end], [fs, fe]) to set E

                # Need to +1 in range  to include the end-point.

                src_phrase = " ".join(srctext[i] for i in range(e_start,e_end+1))

                trg_phrase = " ".join(trgtext[i] for i in range(fs,fe+1))

                # Include more data for later ordering.

                phrases.add(((e_start, e_end+1), src_phrase, trg_phrase))

                fe += 1 # fe++

                # -until fe aligned or out-of-bounds

                if fe in f_aligned or fe == trglen:

                    break

            fs -=1  # fe--

            # -until fs aligned or out-of- bounds

            if fs in f_aligned or fs < 0:

                break

        return phrases



    # Calculate no. of tokens in source and target texts.

    srctext = srctext.split()   # e

    trgtext = trgtext.split()   # f

    srclen = len(srctext)       # len(e)

    trglen = len(trgtext)       # len(f)

    # Keeps an index of which source/target words are aligned.

    e_aligned = [i for i,_ in alignment]

    f_aligned = [j for _,j in alignment]



    bp = set() # set of phrase pairs BP

    # for e start = 1 ... length(e) do

    # Index e_start from 0 to len(e) - 1

    for e_start in range(srclen):

        # for e end = e start ... length(e) do

        # Index e_end from e_start to len(e) - 1

        for e_end in range(e_start, srclen):

            # // find the minimally matching foreign phrase

            # (f start , f end ) = ( length(f), 0 )

            # f_start ∈ [0, len(f) - 1]; f_end ∈ [0, len(f) - 1]

            f_start, f_end = trglen-1 , -1  #  0-based indexing

            # for all (e,f) ∈ A do

            for e,f in alignment:

                # if e start ≤ e ≤ e end then

                if e_start <= e <= e_end:

                    f_start = min(f, f_start)

                    f_end = max(f, f_end)

            # add extract (f start , f end , e start , e end ) to set BP

            phrases = extract(f_start, f_end, e_start, e_end)

            if phrases:

                bp.update(phrases)

    return bp
def print_phrases(phrases):

    # Keep track of translations of each phrase in srctext and its

    # alignement using a dictionary with keys as phrases and values being

    # a list [e_alignement pair, [f_extractions, ...] ]

    dlist = {}

    for p, a, b in phrases:

        if a in dlist:

            dlist[a][1].append(b)

        else:

            dlist[a] = [p, [b]]

            

    # Sort the list of translations based on their length.  Shorter phrases first.

    for v in dlist.values():

        v[1].sort(key=lambda x: len(x))



    # Function to help sort according to book example.

    def ordering(p):

        k,v = p

        return v[0]



    for i, p in enumerate(sorted(dlist.items(), key = ordering), 1):

        k, v = p

        print("({0:2}) {1} {2} — {3}".format( i, v[0], k, " ; ".join(v[1])))
# 1-based indexing.

alignments = [

    [(1,1),(2,2),(3,2),(4,4),(2,3),(5,4),(5,6),(6,3),(7,2)],

    [(7,8)],

    [(i,i) for i in range(1,6)],

]
for A in alignments:

    print('Alignment:', A)

    e_len = max(a[0] for a in A)

    f_len = max(a[1] for a in A) 

    srctext = ' '.join(map(str, range(1, e_len+1)))

    print('srctext:', srctext)

    trgtext = ' '.join(map(str, range(1, f_len+1)))

    print('trgtext:', trgtext)

    phrases = phrase_extraction(srctext, trgtext, A)

    print('Number of phrases:', len(phrases))

    print_phrases(phrases)

    print('-----------------------\n\n')
A = {(1,1),(2,2),(2,3),(2,4),(3,6),(4,7),(5,10),(6,10),(7,8),(8,8),(9,9)}

srctext = 'michael assumes that he will stay in the house'

trgtext = 'michael geht davon aus , dass er im haus bleitbt'
phrases = phrase_extraction(srctext, trgtext, A)
for p, english, german in phrases:

    print(f'{german} ||| {english}')