import numpy

import gzip

from collections import defaultdict

import sys

import numpy

import gzip

from collections import defaultdict

import pandas as pd

import json

import gzip

from tqdm import tqdm_notebook

import Levenshtein

import matplotlib.pyplot as plt

%matplotlib inline



from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/graphicalgenomeapi/CCGG_extension.py", dst = "../working/CCGG_extension.py")



# import all our functions

import CCGG_extension as CCGG
def loadFasta(filename):

    """ Parses a classically formatted and possibly 

        compressed FASTA file into a list of headers 

        and fragment sequences for each sequence contained.

        The resulting sequences are 0-indexed! """

    if (filename.endswith(".gz")):

        fp = gzip.open(filename, 'r')

    else:

        fp = open(filename, 'r')

    # split at headers

    data = fp.read().split('>')

    fp.close()

    # ignore whatever appears before the 1st header

    data.pop(0)     

    headers = []

    sequences = []

    for sequence in data:

        lines = sequence.split('\n')

        headers.append(lines.pop(0))

        sequences.append(''.join(lines))

    return (headers, sequences)



def construct_kmers(filename, k, profiletype):

    '''Construct kmer profile given a reference fasta.

    filename - <str> input reference fastafile

    k - <int> length of kmers

    profiletype - <int> default 0 to query every non-overlapping kmers in the reference genome, 

                                1 to query every possible kmers in the reference genome

    '''

    def contig_name(headers):

        X = []

        for item in headers:

            X.append(item.split(' ')[0])

        return X



    def Construct_nonoverlapping_kmerprofile(X, sequences, k):

        '''Construct kmer profile for every non-overlapping kmers in the linear reference

        X - list of contig name

        sequences - list of reference sequences

        k - length of kmer

        '''

        D = {}

        for i in range(len(sequences)): 

            contig = X[i]

            seq = sequences[i] # 0 offset

            num = len(seq)//k

            for i in range(num):

                D[contig] = D.get(contig, []) + [seq[i*k:(i+1)*k]]

        return D

    

    def Construct_allpossible_kmerprofile(X, sequences, k):

        '''Construct kmer profile for every kmers in the linear reference

        X - list of contig name

        sequences - list of reference sequences

        k - length of kmer

        '''

        D = {}

        for i in range(len(sequences)): 

            contig = X[i]

            seq = sequences[i] # 0 offset

            num = len(seq) - k + 1

            for i in range(num):

                D[contig] = D.get(contig, []) + [seq[i: i+k]]

        return D



    def main(filename, k, profiletype):

        '''Construct kmer profile given a reference fasta.

        filename - <str> input reference fastafile

        k - <int> length of kmers

        profiletype - <int> default 0 to query every non-overlapping kmers in the reference genome, 

                                    1 to query every possible kmers in the reference genome

        '''

        headers, sequences = loadFasta(filename)

        X = contig_name(headers)

        if profiletype == 0:

            D = Construct_nonoverlapping_kmerprofile(X, sequences, k)

        elif profiletype == 1:

            D = Construct_allpossible_kmerprofile(X,sequences, k)

        return D

    

    D = main(filename, k, profiletype)

    return D



# search kmer count

def mapping_position(genome, Kmer_Dict, k):   

    def create_kmer_profile(Kmer_Dict):

        KmerProfile = defaultdict(list)

        for sample,kmers in Kmer_Dict.items():

            for seq in kmers:

                KmerProfile[seq]

                rev = ''.join([{'A':'T','C':'G','G':'C','T':'A'}[base] for base in reversed(seq)]) 

                KmerProfile[rev]

        return KmerProfile



    def mapping(seq, KmerProfile, k):

        for i in range(0, len(seq) - k + 1):

            kmer = seq[i:i+k]



            if 'N' in kmer:

                continue



            c = KmerProfile.get(kmer, -1)



            if isinstance(c, int):

                continue



            KmerProfile[kmer] += [i]

        return KmerProfile



    KmerProfile = create_kmer_profile(Kmer_Dict)

    PositionProfile = mapping(genome, KmerProfile, k)

    return PositionProfile



def create_anchors(kmers, PositionProfile, k):

    """Select Unique kmers as anchor candidates and provide a unique name to each anchor candidates"""

    def unique_kmers(kmers, PositionProfile):

        candidatelist = list(kmers.values())[0]

        candidates = []

        for kmer in candidatelist:

            if len(PositionProfile[kmer]) == 0:

                print(kmer, PositionProfile[kmer])

            krev = ''.join([{'A':'T','C':'G','G':'C','T':'A'}[base] for base in reversed(kmer)]) 

            poslist = PositionProfile[kmer] + PositionProfile[krev]

            if len(poslist) == 1:

                candidates.append((kmer, PositionProfile[kmer][0]))

        return candidates



    candidates = unique_kmers(kmers, PositionProfile)

    anchor_info = numpy.array(candidates)

    return anchor_info



# genome registration

# search kmer count

def genome_registration(genome, anchor_filename, k):   

    def create_kmer_profile(kmerlist):

        KmerProfile = defaultdict(list)

    #         for sample,kmers in Kmer_Dict.iteritems():

        for seq in kmerlist:

            KmerProfile[seq]

            rev = ''.join([{'A':'T','C':'G','G':'C','T':'A'}[base] for base in reversed(seq)]) 

            KmerProfile[rev]

    #         print len(KmerProfile)

        return KmerProfile



    def mapping(seq, KmerProfile, k):

        for i in range(0, len(seq) - k + 1):

            kmer = seq[i:i+k]



            if 'N' in kmer:

                continue



            c = KmerProfile.get(kmer, -1)



            if isinstance(c, int):

                continue



            KmerProfile[kmer] += [i]

        return KmerProfile



    # determine uniqueness

    def unique_kmers(kmerlist, PositionProfile):

        candidates = []

        for kmer in kmerlist:

            if len(PositionProfile[kmer]) > 0:

                krev = ''.join([{'A':'T','C':'G','G':'C','T':'A'}[base] for base in reversed(kmer)]) 

                poslist = PositionProfile[kmer] + PositionProfile[krev]



                if len(poslist) == 1:

                    #print PositionProfile[kmer]

                    candidates.append((kmer, PositionProfile[kmer][0]))

        return candidates



    # determine monotonicity

    def binary_search(arr, val, l, r): 

        if l == r: 

            if arr[l] > val: 

                return l 

            else: 

                return l+1

        if l > r: 

            return l 



        mid = (l+r)/2

        if arr[mid] < val: 

            return binary_search(arr, val, mid+1, r) 

        elif arr[mid] > val: 

            return binary_search(arr, val, l, mid-1) 

        else: 

            return mid 



    # NlogN 

    def efficientDeletionSort(array):

        subsequence_end = [0] # index of the element

        predecessors = [-1] # predecessor index

        for i in range(1,len(array)):

            arr = array[i]

            # can do binary search instead, just skip to make it faster

            if arr > array[subsequence_end[-1]]:

                predecessors += [subsequence_end[-1]]

                subsequence_end += [i]

            else:

                # preform binary search

                minimum_end = [array[j] for j in subsequence_end] # element in current subsequence



                insert_point = binary_search(minimum_end, arr, 0, len(minimum_end)-1)

                if insert_point > 0:

                    predecessors += [subsequence_end[insert_point-1]]

                else:

                    predecessors += [-1]



                if insert_point > len(subsequence_end)-1: # arr == array[subsequence_end[-1]] 

                    subsequence_end += [i]

                elif arr < array[subsequence_end[insert_point]]:

                    subsequence_end[insert_point] = i 



        # backtrack

        pre = subsequence_end[-1]

        listIndex = []

        while pre != -1:

            listIndex.append(pre)

            pre = predecessors[pre]

        listIndex.reverse()

        longest_subsequence = [array[i] for i in listIndex]

        return listIndex, longest_subsequence



    def monotonic_kmers(candidates):

        candidates = numpy.array(candidates)

        poslist = candidates[:,1].astype(int)

        listIndex, longest_subsequence = efficientDeletionSort(poslist)

        monotonic_anchor = candidates[listIndex,:]

        return monotonic_anchor



    anchor_c = numpy.load(anchor_filename)

    kmerlist = anchor_c[:,0]

    KmerProfile = create_kmer_profile(kmerlist)

    PositionProfile = mapping(genome, KmerProfile, k)

    candidates = unique_kmers(kmerlist, PositionProfile)

    final = monotonic_kmers(candidates)

    

    return final





def getsequence_info(genomefile):

    def contig_name(headers):

        X = []

        for item in headers:

            X.append(item.split(' ')[0])

        return X

    

    header, sequence = loadFasta(genomefile)

    samples = contig_name(header)

  

    return header, sequence, samples



def integrate_info(genomefile, anchorfile, k):

    header, sequence, samples = getsequence_info(genomefile)

    anchor_c = numpy.load(anchorfile)

    Anchor_Info = {}

    for i in range(len(header)):

        genome = "+" + sequence[i].upper()

        

        # if not complete genome

        if len(genome) < 10000:

            continue

            

        # if too many ambiguous bases

        basenum = genome.count("A") + genome.count("G") + genome.count('C') + genome.count("T")

        if float(basenum)/len(genome) < 0.9:

            continue

        anchormapping = genome_registration(genome, anchorfile, k)



        Final_Dict= dict(anchormapping) # current mapping



        samplename = samples[i]

        D = {}

        for anchorseq, refpos in anchor_c:

            anchorname = "A%05d" % (int(refpos)/k + 1)

            D[anchorname] = Final_Dict.get(anchorseq, "?")

        Anchor_Info[samplename] = D

    df = pd.DataFrame(Anchor_Info)

    df.to_csv('RegistrationTable.csv')

    return df

    



def writeGraphFasta(filename, input_dict, keylist=["src", "dst"]):

    """Write the given node or edge file as a FASTA file. Overwrites existing file. Will create file if it doesn't exist. 

    def writeFasta(self:<GraphicalGenome>, filename:<str>, input_dict:<dict>, keylist=["src","dst"]:<list[str]>) -> Void



    Parameters:

        filename: <str> - absolute path for the file you wish to write. 

        input_dict: <dict> - Node or Edge dictionary you wish to write. 

        keylist: <list[str]> - list of strings of dictionary keys that you want to ignore during write. 

                This will require you to write these values on your own or just discard them. 

    """

    sorted_keys = sorted(input_dict.keys()) 

    with open(filename, "w+") as fastafile:

        # If iterating through the edges, write the edges in the correctly ordered format

        if (sorted_keys[0][0] == "E"):

            for edge in sorted_keys:

                # If header has not been evaluated, just re-write the header wholesale without any analysis

                if "hdr" in input_dict[edge].keys():

                    line = ">" + edge + ";" + input_dict[edge]["hdr"] + "\n"

                    line += input_dict[edge]["seq"] + "\n"

                    continue

                line = ">" + edge + ";{" 

                # Source

                line += '"src":"' + input_dict[edge]["src"] + '",'

                # Destination

                line += '"dst":"' + input_dict[edge]["dst"] + '"'

                for key in input_dict[edge].keys():

                    if key == "seq":

                        continue

                    if key in keylist:

                        continue

                    line += ',"' + key + '":' + json.dumps(input_dict[edge][key], separators=(",", ":"))

                line += "}\n"

                line += input_dict[edge]["seq"] + "\n"

                fastafile.write(line)

        # If iterating over nodes, just write the nodes normally

        else:

            for i in sorted_keys:

                line = ">" + i + ";"

                obj = {}

                for j in input_dict[i].keys():

                    if j == 'seq':

                        continue

                    obj[j] = input_dict[i][j]

                line += json.dumps(obj, separators=(",", ":"))

                line += "\n" + input_dict[i]['seq'] + "\n"

                fastafile.write(line)



# Dynamically updating the Registration Table

def update_registration_table(genomefile, anchorfile, registrationfile, k):

    '''Updating newly released linear genome in the NCBI Virus Platform to the registration table

    Input: 

    genomefile - <filename> newly released genome collection, sequences should include the assemblies 

    incorporated in the registration table but also contain new genomic sequences

    anchorfile - <filename> numpy file (.npy) of the anchor sequences and their reference coordinates

    registrationfile - <filename> anchor by assemblyID matrix in cvs file format

    k - <int> anchor length

    Output:

    Integration - <dataframe> updated registration table with same row number, larger column number

    '''

    # integrating    

    def get_new_sequences(newfile,registrationfile):

        df = pd.read_csv(registrationfile, index_col= 0)

        new_collection = loadFasta(newfile)



        previous_sample = df.columns

        current_header, current_seq, current_sample = getsequence_info(newfile)

        new_index = [i for i in range(len(current_sample)) if current_sample[i] not in previous_sample]

        new_h = [current_header[i] for i in new_index] 

        new_seq = [current_seq[i] for i in new_index] 

        new_s = [current_sample[i] for i in new_index] 

        return new_h, new_seq, new_s

    

    def get_new_matrix(genomefile, anchorfile, registrationfile, k):

        header,sequence,samples = get_new_sequences(genomefile,registrationfile)

        anchor_c = numpy.load(anchorfile)

        Anchor_Info = {}

        for i in range(len(header)):

            genome = "+" + sequence[i].upper()

            # if not complete genome

            if len(genome) < 10000:

                continue

            # if too many ambiguous bases

            basenum = genome.count("A") + genome.count("G") + genome.count('C') + genome.count("T")

            if float(basenum)/len(genome) < 0.9:

                continue

            anchormapping = genome_registration(genome, anchorfile, k)

            

            Final_Dict= dict(anchormapping) # current mapping



            samplename = samples[i]

            D = {}

            for anchorseq, refpos in anchor_c:

                anchorname = "A%05d" % (int(refpos)/k + 1)

                D[anchorname] = Final_Dict.get(anchorseq, "?")

            Anchor_Info[samplename] = D

        return Anchor_Info



    new_pos = pd.DataFrame(get_new_matrix(genomefile, anchorfile, registrationfile, k))

    current_pos = pd.read_csv(registrationfile,header=0, index_col= 0)



    if len(new_pos) == 0: 

        Integration = current_pos

        print("No Assembly Added")

    else:

        new_pos = new_pos.loc[current_pos.index, :]

        Integration = pd.merge(new_pos, current_pos, left_index=True, right_index=True, how='outer')



    return Integration
# Pangenome Construction



def dynamic_construct(df, exclude_strainlist = []):

    '''Given anchor mapping dictionary, select conserved, unique and monotonically ordered anchors

    Input: 

    df - <dataframe> registration table: anchor candidates by assembly id

    exclude_strainlist - <list> assembly ID that are excluded from the original sets

    Output:

    AnchorList - <DataFrame> anchor by assemblyID matrix, each element is the unique mapping position of the anchor

    '''

    df = df.astype('str')

    strainlist = list(df.columns)

    strainlist = [item for item in strainlist if item not in exclude_strainlist]

    df = df.loc[:,strainlist]

    d = df == '?'

    Anchorlist = df[d.sum(axis = 1) == 0]

    return Anchorlist



# collapse adjacent anchors

def collapsed_adjacent(AnchorDataFrame, k):

    """Given Anchor Table, Collapsed adjacent anchor candidates by only keeping the first and the last anchor in a continuous run

    Input:

    AnchorDataFrame: - <DataFrame>, anchor by assembly

    k - <int> kmer length

    Output:

    AnchorDataFrame: <Dataframe>, anchor by assembly matrix after collapse adjacent anchors

    """

    def find_blocks(Indexlist, k):

        count = 0

        Blocks = []

        for i,s in enumerate(Indexlist):

            if i == 0:

                if Indexlist[i+1] - Indexlist[i] <= k:

                    count += 1

                    start = i

            elif i > 0 and i < len(Indexlist)-1:

                if Indexlist[i] - Indexlist[i-1] > k and Indexlist[i+1] - Indexlist[i] <= k:

                    count +=1

                    start = i

                elif Indexlist[i] - Indexlist[i-1] <= k and Indexlist[i+1] - Indexlist[i] > k:

                    end = i+1

                    Blocks.append((start, end))

            else:

                if Indexlist[i] - Indexlist[i-1] <= k:

                    end = i+1

                    Blocks.append((start, end))

        return count, Blocks





    def sort_by_kmerlength(poslist, k):

        poslist = numpy.array(poslist).astype(int)

        errorindex = []

        count, blocks = find_blocks(poslist, k)

        for s,e in blocks:

            if e - s < 3:

                for i in range(s+1,e):

                    errorindex.append(i)

            else:

                for i in range(s+1,e-1):

                    errorindex.append(i)

        return errorindex



    columnnames = AnchorDataFrame.columns

    for i in columnnames:

        anchors = AnchorDataFrame.index

        poslist = AnchorDataFrame[i].values.astype(int)

        assert sum(sorted(poslist) == poslist) == len(poslist)

        errorindex = sort_by_kmerlength(poslist, k)

        index = [i for i in range(len(anchors)) if i not in errorindex]

        AnchorDataFrame = AnchorDataFrame.iloc[index,:]

        

    return AnchorDataFrame



def create_pangenome(AnchorDataFrame, genomefile,k):

    '''Create pangenome scaffold from Anchor Table and genome assembly

    Input 

       AnchorDataFrame - <Dataframe> Table of anchors and their mapping position

       genomefile - <fasta> filename of the sequence collection

    Output:

    Anchors - <dict> Anchor Information 

    Edges - <dict> Edge Information

    '''

    AnchorDataFrame = collapsed_adjacent(AnchorDataFrame,k)

    

    def genome_info(genomefile):

        header, sequence, samples = getsequence_info(genomefile)

        genome_Dict = {}

        for i in range(len(header)):

            genome = "+" + sequence[i]

            genome_Dict[samples[i]] = genome

        return genome_Dict

    

    def create_nodefile(genome_Dict, AnchorDataFrame,k):

        genome_Dict = genome_info(genomefile)

        Anchors = {}

        refname = 'MN908947'

        for anchor in AnchorDataFrame.index:

            genome = genome_Dict[refname]

            Anchors[anchor] = dict(AnchorDataFrame.loc[anchor])

            pos = Anchors[anchor][refname]

            Anchors[anchor]['seq'] = genome[int(pos):int(pos) + k]  

        return Anchors

    

    def get_all_edge_sequence(AnchorDataFrame, genome_Dict, src, dst):

        Strainlist = AnchorDataFrame.columns

        D = {}

        for strain in Strainlist:

            seq = genome_Dict[strain].upper()

            if src == 'SOURCE':

                for i, s in enumerate(seq):

                    if s!= "N" and s != '+':

                        start = i

                        break

            else:

                start = int(AnchorDataFrame.loc[src, strain]) + k



            if dst == 'SINK':

                end = len(seq)

            else:

                end = int(AnchorDataFrame.loc[dst, strain])



            edge_seq = seq[start:end]

            D[strain] = edge_seq



        return D

    

    def get_edge_info(src,dst, AnchorDataFrame, genome_Dict):

        D = get_all_edge_sequence(AnchorDataFrame, genome_Dict, src, dst)

        Strain = Anchorlist.columns

        Edge_D = {}

        index = 0

        my_k = 0

        while my_k in range(len(Strain)):

            index += 1

            if src != "SOURCE":

                edgename = 'E%08d' % (int(src[1:])*1000 + index)

            else:

                edgename = 'S%08d' % (index)

            Edge_D[edgename] = {}

            strain = Strain[my_k] 

            Name = []        

            Name.append(strain)



            if len(Strain) - my_k>1:

                for j in range(my_k+1, len(Strain)):

                    strain1 = Strain[j]

                    if D[strain] == D[strain1]:

                        Name.append(strain1)

            Strain = [x for x in Strain if x not in Name]



            Edge_D[edgename]['seq'] = D[strain]

            Edge_D[edgename]['strain'] = sorted(Name)

            Edge_D[edgename]['src'] = src

            Edge_D[edgename]['dst'] = dst



        return Edge_D    

    

    def create_edgefile(genome_Dict, AnchorDataFrame, k):

        nodelist = ["SOURCE"] + list(AnchorDataFrame.index) + ['SINK']

        Edge_info = {}

        total = 0

        for i in range(len(nodelist) - 1):

            src = nodelist[i]

            dst = nodelist[i + 1]

            Edges = get_edge_info(src,dst, AnchorDataFrame, genome_Dict)

            total += len(Edges)

            Edge_info.update(Edges)    

        return Edge_info

    

    genome_Dict = genome_info(genomefile)

    Anchors = create_nodefile(genome_Dict,AnchorDataFrame,k)

    Edges = create_edgefile(genome_Dict, AnchorDataFrame, k)

    

    return Anchors, Edges
# add annotation



# add genome feature

# [seqid, source, type, start, end, score, strand, phase, attribute]

def get_gff_data(filename):

    if filename.endswith('.gz'):

        with gzip.open(filename, 'rb') as fp:

            data = fp.readlines()

    else:

        with open(filename, 'rb') as fp:

            data = fp.readlines()

    return data



def parse_feature(data):

    Parser = {}

    count = 0

    for line in data:

        values = line.decode().split('\t')

        if len(values) < 9:

            continue

        start, end = int(values[3]), int(values[4])

        # parse attributes

        attributes = values[8]

        Att = {}

        attributes_info = attributes.split(';')

        for item in attributes_info:

            k, v = item.split('=')

            Att[k] = v

        #assert "ID" in Att.keys()

        # build in   

        name = Att["ID"]

        Parser[name] = {}

        Parser[name]["start"] = start

        Parser[name]['end'] = end

        Parser[name]['att'] = Att

        Parser[name]['strand'] = values[6]

        Parser[name]['type'] = values[2]

    return Parser



def annotatefeature(graph, filename,k):    

    def write_cigar(g_start, g_end, i_start, i_end):

        ''' input genestart, geneend, itemstart, item_end Gene include both ends [], items include the start [) '''

        g_start = int(g_start)

        g_end = int(g_end)

        i_start = int(i_start)

        i_end = int(i_end)

        if i_start >= g_start and i_end - 1 <= g_end:

            Cigar = str(i_end - i_start)+'M'



        elif i_start < g_start and i_end - 1 <= g_end:

            miss = g_start - i_start

            Cigar = str(miss)+'S'+str(i_end-1-g_start+1)+'M'



        elif i_start < g_start and i_end -1 > g_end: # gene inside a Node

            miss = g_start - i_start

            Cigar = str(miss)+'S'+str(g_end - g_start + 1)+'M' + str(i_end - 1 - g_end)+'S'



        elif i_end - 1 > g_end:

            miss = i_end -1 - g_end

            Cigar = str(g_end - i_start + 1)+'M'+str(miss)+'S'



        return Cigar





    def find_itempos(Graph, item, k):

        '''Input graph item, return start and end pos [istart, iend), end pos not included'''

        if item.startswith('A'):

            i_start = int(Graph.nodes[item]['MN908947'])

            i_end = i_start + k

        elif item.startswith('F') or item.startswith('B') or item == 'SOURCE' or item == 'SINK':

            return 

        else:

            snode = Graph.edges[item]['src']

            if snode == 'SOURCE':

                i_start = 1

            else:

                i_start = int(Graph.nodes[snode]['MN908947']) + k



            i_end = i_start + len(Graph.edges[item]['seq'])



        return (i_start, i_end)



    def addannotation(Graph, Feature_info,k):

        featurelist = Feature_info.keys()

        for g_ID in featurelist:

            D = Feature_info[g_ID]

            g_start = int(D['start'])

            g_end = int(D['end'])

            strand = D['strand']

            key = D['type']

            sanchor, eanchor = Graph.boundingAnchors('MN908947', g_start, g_end)

            itemlist = Graph.tracePath('MN908947', sanchor, eanchor)



            for item in itemlist:

                if item == 'SOURCE' or item == 'SINK':

                    continue

                i_start, i_end = find_itempos(graph, item,k)

                if i_end <= g_start or i_start > g_end:

                    continue

                    

                cigar = write_cigar(g_start, g_end, i_start, i_end)

#                 if cigar == '0M':

#                     print( cigar, item,g_start, g_end, i_start, i_end, len(Graph.edges[item]['seq']))

                

                if item.startswith('A'):

                    annote = '|'.join((g_ID, strand, cigar))

                    if annote in Graph.nodes[item].get(key, []):

                        continue

                    Graph.nodes[item][key] = Graph.nodes[item].get(key, []) + ['|'.join((g_ID, strand, cigar))]

                else:

                    annote = '|'.join((g_ID, strand, cigar))

                    if annote in Graph.edges[item].get(key, []):

                        continue

                    Graph.edges[item][key] = Graph.edges[item].get(key, []) + ['|'.join((g_ID, strand, cigar))]

        return Graph

    

    data = get_gff_data(filename)

    feature_info = parse_feature(data)

    graph = addannotation(graph, feature_info,k)

    return graph



# add variants

def addvariants(Graph, maxlength,  refstrain = "MN908947"):

    def makeCigar(seq, ref):

        if (len(seq) > 16384) or (len(ref) > 16384):

            rmid = len(ref)/2        

            smid = len(seq)/2

            prox = makeCigar(seq[:smid],ref[:rmid])

            dist = makeCigar(seq[smid:],ref[rmid:])

            return prox+dist

        ops = Levenshtein.editops(seq,ref)

        code = ['=' for i in range(len(seq))]

        offset = 0

        for op, si, di in ops:

            if (op == "replace"):

                code[si+offset] = 'X'

            elif (op == "insert"):

                code.insert(si+offset,'D')

                offset += 1

            elif (op == "delete"):

                code[si+offset] = 'I'# LM: fixed bug here 2019-04-15

        cigar = ''

        count = 1

        prev = code[0]

        for c in code[1:]:

            if (c == prev):

                count += 1

            else:

                cigar += "%d%c" % (count, prev)

                count = 1

                prev = c

        cigar += "%d%c" % (count, prev)

        return cigar



    # check the length of the cigar string is valid

    def search_valid_size(variants,seq):

        ref_length,alt_length = 0,0

        for v in variants:

            base, op = v

            base = int(base)

            if op == '=' or op == 'X':

                ref_length += base

                alt_length += base

            elif op == 'I':

                alt_length += base

            elif op == 'D':

                ref_length += base



        return len(seq) == alt_length 



    # check the length of the cigar string is valid

    def valid_size(variants,seq,reference):

        ref_length,alt_length = 0,0

        for v in variants:

            base, op = v

            base = int(base)

            if op == '=' or op == 'X':

                ref_length += base

                alt_length += base

            elif op == 'I':

                alt_length += base

            elif op == 'D':

                ref_length += base



        assert len(seq) == alt_length 

        assert len(reference) == ref_length



    def valid_eqaul_and_mismatch(variants,seq,reference):

        ref_pos,alt_pos = 0,0

        for v in variants:

            base,op = v

            base = int(base)

            if op == '=':

                assert seq[alt_pos:alt_pos+base] == reference[ref_pos:ref_pos+base]

                ref_pos += base

                alt_pos += base

            elif op == 'X':

                assert seq[alt_pos:alt_pos+base] != reference[ref_pos:ref_pos+base]

                ref_pos += base

                alt_pos += base

            elif op == 'I':

                alt_pos += base

            elif op == 'D':

                ref_pos += base



    #assert len(reference) == ref_length

    def split(variant):

        splitVariants = []

        previous = 0

        for i in range(len(variant)):

            v = variant[i]

            if v.isdigit():

                continue

            else:

                splitVariants += [variant[previous:i]]

                splitVariants += [v]

                previous = i + 1



        numbers, types = [],[]

        for i in range(len(splitVariants)):

            v = splitVariants[i]

            if i %2 == 0:

                numbers += [v]

            else:

                types += [v]



        variant = []

        for i in range(len(numbers)):

            variant += [(numbers[i],types[i])]

        return variant



    def findBedge(Graph, src, dst, refstrain):

        edgelist = Graph.outgoing[src]

        for edge in edgelist:

            sequence = Graph.edges[edge]['seq'].upper()

            strain = Graph.edges[edge]['strain']

            if refstrain in strain:

                B_dst = Graph.edges[edge]['dst']

                if B_dst == dst:

                    return edge

                else:

                    return ""

        else:

            return ''



    # mask Ns on the alternative path

    def maskNs(seq):

        seq = seq.upper()

        newseq = ''

        for i in seq:

            if i not in "AGCT":

                newseq += i.lower()

            else:

                newseq += i

        return newseq

    

    def add_variants(Graph, maxlength, refstrain, gapopen = False):

        Nodelist = sorted(Graph.nodes.keys())

        Nodelist = ["SOURCE"] + Nodelist

        for node in Nodelist:

            sanchor = node

            eanchor = Graph.nextAnchor(sanchor)

            edgelist = Graph.outgoing[sanchor]

            ref_edge = findBedge(Graph, sanchor, eanchor, refstrain)



            # fix redundant Ns on SOURCE

            if sanchor == "SOURCE":

                seq = Graph.edges[ref_edge]['seq'].upper()

                for i, s in enumerate(seq):

                    if s!= "N":

                        start = i

                        break

                ref_seq = seq[start:]

                ref_length = len(ref_seq)

                Graph.edges[ref_edge]['seq'] = ref_seq  

            # fix redundant Ns on SINK

            elif eanchor == "SINK":

                seq = [i for i in reversed(Graph.edges[ref_edge]['seq'].upper())]

                for i, s in enumerate(seq):

                    if s!= "N":

                        start = i

                        break         

                ref_seq = seq[start:]

                ref_seq = ''.join([i for i in reversed(ref_seq)])

                ref_length = len(ref_seq)

                Graph.edges[ref_edge]['seq'] =  ref_seq

            else:

                ref_seq = Graph.edges[ref_edge]['seq'].upper()

                ref_length = len(ref_seq)







            if ref_length > maxlength:

                continue



            for alt_edge in edgelist:

                if alt_edge == ref_edge:

                    Graph.edges[ref_edge]['variants'] = str(ref_length) + '='

                    continue



                alt_seq = maskNs(Graph.edges[alt_edge]['seq']) # mask Ns

                alt_length = len(alt_seq)



                if alt_length > maxlength:

                    continue



                if gapopen:

                    delta = abs(alt_length - ref_length)

                    if delta > 100:

                        if delta > alt_length or delta > ref_length:

                            cigar = GapOpenAligner(ref_seq, alt_seq)

                            variants = split(cigar)

                            valid_size(variants, alt_seq, ref_seq)

                            valid_eqaul_and_mismatch(variants, alt_seq, ref_seq)

                        else:

                            cigar = makeCigar(alt_seq, ref_seq)

                            variants = split(cigar)

                            valid_size(variants, alt_seq, ref_seq)

                            valid_eqaul_and_mismatch(variants, alt_seq, ref_seq)

                    else:

                        cigar = makeCigar(alt_seq, ref_seq)

                        variants = split(cigar)

                        valid_size(variants, alt_seq, ref_seq)

                        valid_eqaul_and_mismatch(variants, alt_seq, ref_seq)

                else:

                    cigar = makeCigar(alt_seq, ref_seq)

                    variants = split(cigar)

                    valid_size(variants, alt_seq, ref_seq)

                    valid_eqaul_and_mismatch(variants, alt_seq, ref_seq)



                Graph.edges[alt_edge]['variants'] = cigar

    

    add_variants(Graph, maxlength, refstrain, gapopen = False)

    return Graph

# Run

filename1 = '/kaggle/input/sarscov2-reference-genbackmn908947/GCA_009858895.3_ASM985889v3_genomic.fna'

header, seq = loadFasta(filename1)

genome = "+" + seq[0].upper()



# Select Valid k

for k in range(5,12):

    kmers = construct_kmers(filename1, k, profiletype = 0)

    print( "kmerlength:", k, ",non-overlapping kmer num:", len(kmers['MN908947.3']), 

          ",kmer Number:", len(seq[0]) - k + 1, "all possible kmer Number:", 4 ** k)

print("Minimal k = ", 8)

# Select anchor candidates

k = 11

%time kmers = construct_kmers(filename1, k, profiletype = 0)

%time PositionProfile = mapping_position(genome, kmers, k)

%time anchorcandidates = create_anchors(kmers, PositionProfile, k)

numpy.save("anchorcandidates",anchorcandidates)

print("Number of Anchor Candidates:", len(anchorcandidates))



anchorcandidates
# Genome Registration

genomefile = '../input/graphicalgenomeapi/NCBI_567.fasta'

anchorfile = "../working/anchorcandidates.npy"

k = 11

%time registration_table = integrate_info(genomefile, anchorfile, k)

#header, sequence = loadFasta(genomefile)
# Pangenome Construction

#genomefile = '../input/graphicalgenomeapi/NCBI_567.fasta'

#registration_table = pd.read_csv("RegistrationTable.csv", index_col = 0)

Anchorlist = dynamic_construct(registration_table)

print(Anchorlist.shape)#Anchorlist1.shape

%time Anchor_Dict,Edge_Dict = create_pangenome(Anchorlist, genomefile, 11)

writeGraphFasta("Cov_Nodefile.fa",Anchor_Dict)

writeGraphFasta("Cov_Edgefile.fa",Edge_Dict)
graph = CCGG.GraphicalGenome("Cov_Nodefile.fa", "Cov_Edgefile.fa", nnamelen= 6, enamelen=9)

%time graph = addvariants(graph, 3000)



annotationfile = "../input/graphicalgenomeapi/annotation.gff"

%time graph = annotatefeature(graph, annotationfile,11)



writeGraphFasta("Cov_Nodefile.fa",graph.nodes)

writeGraphFasta("Cov_Edgefile.fa",graph.edges)

# validation

graph = CCGG.GraphicalGenome("Cov_Nodefile.fa", "Cov_Edgefile.fa", nnamelen= 6, enamelen=9)



def Validate_sequence(Graph, genome_Dict, strain, path = 0):

    conseq = Graph.reconstructSequence(strain, path)

    lineargenome = genome_Dict[strain][1:].upper()

    return conseq.upper() == lineargenome.upper()



strainlist = Anchorlist.columns

print( len(strainlist))



def genome_info(genomefile):

    header, sequence, samples = getsequence_info(genomefile)

    genome_Dict = {}

    for i in range(len(header)):

        genome = "+" + sequence[i]

        genome_Dict[samples[i]] = genome

    return genome_Dict



genome_Dict = genome_info(genomefile)

print(len(genome_Dict))

for strain in strainlist:

    assert Validate_sequence(graph, genome_Dict, strain)

print("PASS")
# Graphical Genome Statistics



graph = CCGG.GraphicalGenome("Cov_Nodefile.fa", "Cov_Edgefile.fa", nnamelen= 6, enamelen=9)

refstrain = 'MN908947'

def getfounders(Graph):

    '''Input graph, return a list of assembly IDs incorporated in the graphical genome'''

    edgelist = Graph.outgoing['SOURCE']

    founder = []

    for edge in edgelist:

        founder += Graph.edges[edge]['strain']

    return founder



# Basic statistics

founders = getfounders(graph)

print( "Number of linear genome", len(founders))

totalsize = 0

nodelist = graph.nodes.keys()

edgelist = graph.edges.keys()

print( "Anchor Num", len(nodelist))

print( 'Edge Num', len(edgelist))



totalsize = 0

for edge in edgelist:

    totalsize += len(graph.edges[edge]['seq'])

for node in nodelist:

    totalsize += len(graph.nodes[node]['seq'])

print( "Pangenome Size", totalsize)



originalsize = 0

genome_Dict = genome_info(genomefile)

for strain in founders:

    originalsize += len(genome_Dict[strain]) - 1 # delete the "+" sign

print("Original Size", originalsize)

print( "Compress Percentage", float(totalsize)/originalsize)

# Anchor Distribution

refseq = graph.reconstructSequence(refstrain,path=0)

intervallength = 1000

nodelist = sorted(graph.nodes.keys())

Anchorpos = [int(graph.nodes[anchor][refstrain]) for anchor in nodelist]

y,x = numpy.histogram(Anchorpos, bins = len(refseq)//intervallength, range = [1, len(refseq) + 1])

fig = plt.figure(figsize = [10,5])

plt.plot(x[1:],y, color='r', linestyle='-',linewidth = 1.5)

#plt.xlim([1,len(refseq)+1])

#plt.xticks(numpy.arange(1, len(refseq)+intervallength, step= intervallength*2) ,rotation = -60) 

plt.xticks(x.astype(int) ,rotation = -60,fontsize = 8) 



plt.xlabel('Genome Position / 1Kb', fontsize = 12)

plt.ylabel('Num of Anchors', fontsize = 12)

plt.title("Anchor Distribution", fontsize = 20)

plt.show()

plt.savefig("AnchorDistribution")

# Edge statistics

founders = getfounders(graph)

edgenum = numpy.zeros(len(founders)) # initialize a vector for edge statistics



# how many edges within each gap

node = "SOURCE"

while node != "SINK":

    edgelist = graph.outgoing[node]

    edgenum[len(edgelist)] += 1

    node = graph.nextAnchor(node)

    

stat = edgenum.astype(int)

HapNum = numpy.where(stat != 0)[0]



plt.figure(figsize = [10,6])

plt.bar(range(len(HapNum)), stat[HapNum])

plt.xticks(range(len(HapNum)), HapNum, fontsize = 15)

plt.xlabel('Edge Num', fontsize = 20)

plt.ylabel("Number of Gaps", fontsize = 20)

plt.show()

plt.savefig("EdgeNumberDistribution")



# What's the genomic fractions for each type of gaps (with 1,2...n haplotypes)?

HapNums = numpy.where(stat != 0)[0] # all possible hap num in graph

print( HapNums)

Haplotype_Fraction = numpy.zeros(len(HapNums)) # edge number between two anchors

k = 11 # length of anchors



# counting the number of reference bases within each gap

node = "SOURCE"

while node != "SINK":

    edgelist = graph.outgoing[node]

    edge_num = len(edgelist)

    if node.startswith('A'):

        Haplotype_Fraction[numpy.where(HapNums == 1)[0][0]] += k

    for edge in edgelist:

        strain = graph.edges[edge]['strain']

        if refstrain in strain:

            Bseq = len(graph.edges[edge]['seq'])

            break

    Haplotype_Fraction[numpy.where(HapNums == edge_num)[0][0]] += Bseq

    

    node = graph.nextAnchor(node)



# print(Haplotype_Fraction)

# print(sum(Haplotype_Fraction))



fig = plt.figure(figsize = [10,8])

plt.bar(range(len(HapNums)), Haplotype_Fraction/float(sum(Haplotype_Fraction)))

plt.xticks(range(len(HapNums)),HapNums,fontsize = 12)

plt.xlabel('Edge Num', fontsize = 20)

plt.ylabel("Genomic Fraction",fontsize = 20)

plt.show()

plt.savefig("EdgeNum_vs_GenomicFraction.png")

# anchor distribution versus outgoing edge number

anchor_edge = []

node = "SOURCE"

while node != "SINK":

    node = graph.nextAnchor(node)

    if node == "SINK":

        continue

    pos = int(graph.nodes[node][refstrain])

    hap = len(graph.outgoing[node])

    anchor_edge.append((pos,hap))

anchoredgeinfo = numpy.array(anchor_edge)[:-1, :]

plt.figure(figsize=[10,8])

#print( max(anchoredgeinfo[:,1]), numpy.argmax(anchoredgeinfo[:,1]), anchoredgeinfo[472,:])

plt.bar(anchoredgeinfo[:,0], anchoredgeinfo[:,1], width=50)

plt.xlabel("Anchor Position", fontsize = 10)

plt.ylabel("Outgoing Edge Num", fontsize = 10)

plt.title('Anchor Distribution and Edge Number within Gaps')

plt.savefig("AnchorDist_vs_edgeNum.png")

# Whole genome Haplotype Distribution for each 500bp region

def get_gene_interval(annotationfilename):

    data=get_gff_data(annotationfilename)

    Parser = parse_feature(data)

    gene_d = {}

    interval = []

    for name,D in Parser.items():

        if D['type'] == 'gene':

            Name = D['att']['Name']

            gene_d[Name] = {}

            gene_d[Name]['start'] = D['start']

            gene_d[Name]['end'] = D['end']

            interval.append((D['start'], D['end'], Name))

    return interval



interval = get_gene_interval(annotationfile)





intervallength = 500

n = len(refseq)//intervallength

#print n, len(refseq)



HapNum = []

# get Hap Num

for i in range(n):

    start = intervallength * i + 1

    end = start + intervallength

    sa, ea = graph.boundingAnchors(refstrain, start, end)

    path = graph.getSubPath(sa,graph.prevAnchor(ea),init_strain= set(strainlist))

    HapNum.append(len(path))

    

path = graph.getSubPath(graph.prevAnchor(ea), 'SINK',init_strain= set(strainlist))

HapNum.append(len(path))

print(HapNum)

print( len(HapNum), max(HapNum[2:-1]))



# plot

# haplotype plot

fig = plt.figure(figsize = [20,10])

plt.plot(range(1,len(refseq), intervallength)[1:-2],HapNum[2:-1], linestyle='-',linewidth = 1.5)

# overlay genes

colors = ['r','g','b','k','grey','orange','purple','pink','steelblue','darkblue','darkred']

#plt.axhline(y=20, xmin=0, xmax= 1, color = 'red', linestyle='-', linewidth=5)

i = 0

interval = sorted(interval)

genenames = []

for s, e, name in interval:

    plt.axhline(y=max(HapNum[2:-1])+1+(i) + 0.5, xmin=float(s)/(len(refseq)+intervallength), xmax= float(e)/(len(refseq)+intervallength), 

                color = colors[i], linestyle='-', linewidth=2)

    genenames.append(name)

    if i < 2:

        plt.text(s + (e-s)/2, max(HapNum[2:-1])+1+(i) +1, name, va='center', ha='center', fontsize = 15)

    else:        

        plt.text(s + (e-s)/2, max(HapNum[2:-1])+1+(i) +1, name, va='center', ha='center', fontsize = 10)    

    i = i + 1



plt.legend(["HaplotypeNum"] + genenames, loc = 'best', fancybox=True, shadow=False, fontsize = 13)# bbox_to_anchor=(1, .95),



    

plt.xlim([1,len(refseq)+1])

plt.ylim([0,max(HapNum[2:-1])+1+(i)+0.8]) # exclude the fist and last bins



plt.xticks(numpy.arange(1001, len(refseq)+intervallength, step= intervallength * 2) ,rotation = -30, fontsize = 15) 

plt.yticks(numpy.arange(1, max(HapNum[2:-1]), step= 2),fontsize = 10) 



plt.xlabel('Genome Position/%sbp' % str(intervallength), fontsize = 20)

plt.ylabel('Num of Paths', fontsize = 20)

plt.title("Haplotype Distribution", fontsize = 20)

plt.show()

fig.savefig("HaplotypeDistribution.png")
# How many distinct versions for each gene?

def anchor_coordinates_exchanging(Graph, s, e, refstrain):

    '''Given a list of intervals on the reference genome, mapping the start and the end coordinates to parallel edges, 

    return all possible distinct sequences through graph traversal

    Input:

    Graph - <class> Graphical Genome

    s - <int> start coordinate

    e - <int> end coordinate

    refstrain - <str> the Assembly ID of the reference genome

    

    Output:

    Haplotype - <dict> A dictionary of distinct sequence versions and a list of assemblies that share the pattern

    '''

    def find_offset_on_path(Graph, itemlist, entity, offset, k = 11):

        prevoff = 0

        for item in itemlist:

            if item == entity:

                path_coord = prevoff + offset

                return path_coord

            else:

                if item.startswith('A'):

                    prevoff += k

                elif item.startswith('F') or item.startswith('B') or item == 'SOURCE' or item == 'SINK':

                    prevoff += 0

                else:

                    prevoff += len(Graph.edges[item]['seq'])





    def offsetexchange(cigar, B_offset):

        alt_i = 0

        ref_i = 0

        for i, s in enumerate(cigar):

            if ref_i == B_offset:

                return alt_i   

            if s == '=':

                alt_i += 1

                ref_i += 1



            if s == 'I':

                alt_i += 1

            if s == 'D':

                ref_i += 1

            if s == 'X':

                alt_i += 1

                ref_i += 1



    def get_all_altoffset(graph, s, refstrain):

        Alt = []

        item,bpos = graph.linear_to_entityoffset(s, refstrain)



        # if start from anchor

        if item.startswith('A'):

            Alt.append((item, bpos))

        else:

        # start from edge

            src = graph.edges[item]['src']

            edgelist = graph.outgoing[src]



            for edge in edgelist:

                strain = graph.edges[edge]['strain']

                cigar = graph.processCigar(graph.edges[edge]['variants'])

                alt_pos = offsetexchange(cigar,bpos)

                Alt.append((edge,alt_pos))

        return Alt



    def get_haplotype_sequence(graph, s, e, refstrain):

        StartPos = get_all_altoffset(graph, s, refstrain)

        EndPos = get_all_altoffset(graph, e, refstrain)



        start, end = graph.boundingAnchors(refstrain, s, e) # Covgenome

        Path = graph.getSubPath(start, end, init_strain= set(founders))

        Haplotype = {}

        for itemlist, strainlist in Path:

            for item, pos in StartPos:

                if item in itemlist:

                    pathstart = find_offset_on_path(graph, itemlist, item, pos, k = 11)

                    break



            for item, pos in EndPos:

                if item in itemlist:

                    pathend = find_offset_on_path(graph, itemlist, item, pos, k = 11)

                    break

            seq = graph.findsequence(itemlist,countinganchor=True)[pathstart:pathend+1]

            Haplotype[seq] = Haplotype.get(seq, []) + list(strainlist)

        return Haplotype



    start, end = Graph.boundingAnchors(refstrain, s, e) # Covgenome

    H = get_haplotype_sequence(Graph, s, e, refstrain)

    

    return H



Geneinfo = []

for s, e, name in interval:

    start, end = graph.boundingAnchors(refstrain, s,e)

    Hap = anchor_coordinates_exchanging(graph, s, e, refstrain)

    Geneinfo.append((name,s,e, e-s, start, end, len(Hap), len(Hap)/float(e-s)*1000))

Geneinfo = pd.DataFrame(Geneinfo, columns = ['genename', 'refstart','refend', 'genelength', 

                                             'startanchor','endanchor','HapNum', "HapNum/kb"])

Geneinfo.sort_values(by=['HapNum/kb'])
def get_variant_position(cigar):

    '''helper function

    identify variants given an expanded cigar string

    return the offset on the alternative edge and the mapping offset on reference path

    '''

    ref_pos = []

    alt_pos = []

    alt_i = 0

    ref_i = 0

    for i, s in enumerate(cigar):

        if s == 'I':

            if ref_i > 0:

                ref_pos.append(ref_i-1)

            else:

                ref_pos.append(ref_i)    

            alt_pos.append(alt_i)

            alt_i += 1

        if s == 'D':

            ref_pos.append(ref_i)

            if alt_i > 0:

                alt_pos.append(alt_i-1)

            else:

                alt_pos.append(alt_i)

            ref_i += 1

            

        if s == 'X':

            ref_pos.append(ref_i)

            alt_pos.append(alt_i)

            alt_i += 1

            ref_i += 1

            

        if s == '=':

            alt_i += 1

            ref_i += 1



    return ref_pos, alt_pos



def find_allVar(graph, refstrain, founder, k):

    '''find all variants, record variants coordinates and its mapping coordinates on the reference genome'''

    Var = {} # Var[refpos]['edgename'] = ['A']

    node = "SOURCE"

    while node != "SINK":

        edgelist = graph.outgoing[node]

        

        for edge in edgelist:

            if refstrain in graph.edges[edge]['strain']:

                refseq = graph.edges[edge]['seq']

                refedge = edge

                break

                

        for edge in edgelist:

            cigar = graph.edges[edge]['variants']

            if node != "SOURCE":

                refstart = int(graph.Nodecoordinates(node, strainlist = founder)[refstrain]) + k

            else:

                refstart = int(graph.Nodecoordinates(node, strainlist = founder)[refstrain])

            

            refpos, altpos = get_variant_position(graph.processCigar(cigar))

            #print refpos, altpos

            alt_seq = graph.edges[edge]['seq']

            for i, rp in enumerate(refpos):

                pos = refstart + rp

                Var[pos] = Var.get(pos, {})

                Var[pos][edge] = Var[pos].get(edge, {})                

                Var[pos][edge]['base'] = Var[pos][edge].get('base', "") + alt_seq[altpos[i]]

                Var[pos][edge]['altoffset'] = Var[pos][edge].get('altoffset', []) + [altpos[i]]

                Var[pos][refedge] = Var[pos].get(refedge, {})

                Var[pos][refedge]['base'] = refseq[refpos[i]]

        node = graph.nextAnchor(node)    #print cigar, refpos, altpos

    return Var



def coordinate_exchange(Graph, coord, alter_strain, refstrain, k):

    '''Given a linear coordinate on the reference path, return the mapping coordinates on alternative genome'''

    def find_offset_on_path(Graph, itemlist, entity, offset):

        prevoff = 0

        for item in itemlist:

            if item == entity:

                path_coord = prevoff + offset

                return path_coord

            else:

                if item.startswith('A'):

                    prevoff += k

                elif item.startswith('F') or item.startswith('B') or item == 'SOURCE' or item == 'SINK':

                    prevoff += 0

                else:

                    prevoff += len(Graph.edges[item]['seq'])



    def completecigar(Graph, edgelist, kmer = k):

        cigar = ''

        for edge in edgelist:

            if edge.startswith('B') or edge.startswith('F') or edge == 'SOURCE' or edge == 'SINK':

                continue

            if edge.startswith('A'):

                cigar += str(kmer) + '='

                continue

            if 'variants' in Graph.edges[edge].keys():

                cigar += Graph.edges[edge]['variants']

            else:

                #print edge

                return False

        else:

            return cigar



    def findsequence(Graph, pathlist):

        seq = ''

        for item in pathlist:

            if item.startswith('A'):

                seq += '' # do not count anchor length

            elif item.startswith('L') or item.startswith('E')or item.startswith('K'):

                seq += Graph.edges[item]['seq']

            elif item.startswith('S') and item != "SOURCE" and item != 'SINK':

                seq += Graph.edges[item]['seq']

            else:

                seq += ''

        return seq

    

    def offsetexchange(cigar, B_offset):

        alt_i = 0

        ref_i = 0

        for i, s in enumerate(cigar):

            if ref_i == B_offset:

                return alt_i   

            if s == '=':

                alt_i += 1

                ref_i += 1



            if s == 'I':

                alt_i += 1

            if s == 'D':

                ref_i += 1

            if s == 'X':

                alt_i += 1

                ref_i += 1

            

    def Complete_Parallel_edge_offset_exchange(Graph, coord, alter_strain, founderlist, refstrain, output = 0):

        """Alignment-based entity offset exchange

        (only apply to parallel edge, edge share the same src and dst, offset exchange in Path scale are not finished yet)

        Given entity+offset on B path, return the position on the alternative strain

        If alignment results are not applicable, return None



        Parameters:

            entity: <str> - Graph entity on B path

            offset: <int> - offset on the entity

            alter_strain: <str> - strain attributes for the target alternative path position

            strain: <default> "B" path, when multi-alignment cigar are added, this could be further implemented

        """

        entity, offset = Graph.linear_to_entityoffset(coord, refstrain)



        if entity.startswith('A'):

            alter_coord = Graph.Nodecoordinates(entity, strainlist=founderlist)[alter_strain] + offset

            return alter_coord, offset

        else:

            src = Graph.edges[entity]['src']    

            if src.startswith('A') or src == 'SOURCE':

                s_anchor = src

            else:

                s_anchor = Graph.prevAnchor(src)

            d_anchor = Graph.nextAnchor(s_anchor)



            # construct path

            Path = Graph.getSubPath(s_anchor, d_anchor, init_strain=founderlist)

            for itemlist, strain in Path:

                if refstrain in strain and alter_strain in strain:

                    path_coord = find_offset_on_path(Graph, itemlist, entity, offset)

                    alter_coord = Graph.Nodecoordinates(s_anchor, strainlist = founderlist)[alter_strain] + path_coord

                    return alter_coord, path_coord

                

                elif refstrain in strain:

                    B_offset = find_offset_on_path(Graph, itemlist, entity, offset)

                    

                elif alter_strain in strain:

                    c_cigar = completecigar(Graph, itemlist)

                    if c_cigar == False: # not applicable in covid19 pangenome

                        for il, bstrain in Path:

                            if refstrain in bstrain:

                                ref_seq = Graph.findsequence(il, countinganchor=True)

                                break

                        alt_seq = Graph.findsequence(itemlist, countinganchor=True)

                        cigar = makeCigar(alt_seq, ref_seq)

                        cigar = Graph.processCigar(cigar)

                    else:

                        cigar = Graph.processCigar(c_cigar)



            path_coord = offsetexchange(cigar, B_offset)

            #print cigar, len(cigar),B_offset

            alter_coord = Graph.Nodecoordinates(s_anchor, strainlist=founderlist)[alter_strain] + path_coord

            return alter_coord, path_coord

        

    founderlist = []

    for edge in Graph.outgoing["SOURCE"]:

        founderlist += Graph.edges[edge]['strain']

    

    alt_coord, path_coord = Complete_Parallel_edge_offset_exchange(Graph, coord, alter_strain, set(founderlist), refstrain)

    

    return alt_coord, path_coord



# collapse adjacent 

def collapsed_adjacent(Hap_Dict, step):

    '''Given the dictionary of Variants, collapsed the allels on a continuous run'''

    k = step

    def find_blocks(Indexlist, k):

        count = 0

        Blocks = []

        for i,s in enumerate(Indexlist):

            if i == 0:

                if Indexlist[i+1] - Indexlist[i] <= k:

                    count += 1

                    start = i

            elif i > 0 and i < len(Indexlist)-1:

                if Indexlist[i] - Indexlist[i-1] > k and Indexlist[i+1] - Indexlist[i] <= k:

                    count +=1

                    start = i

                elif Indexlist[i] - Indexlist[i-1] <= k and Indexlist[i+1] - Indexlist[i] > k:

                    end = i+1

                    Blocks.append((start, end))

            else:

                if Indexlist[i] - Indexlist[i-1] <= k:

                    end = i+1

                    Blocks.append((start, end))

        return count, Blocks





    def sort_by_kmerlength(poslist, k):

        poslist = numpy.array(poslist).astype(int)

        errorindex = []

        count, blocks = find_blocks(poslist, k)

        pos_inblocks = []

        for s,e in blocks:

            pos_inblocks.append((poslist[s], poslist[e-1]))

            

        for s,e in blocks:

            for i in range(s,e):

                errorindex.append(i)

                

        return errorindex, pos_inblocks



    RefIndexlist = sorted(Hap_Dict.keys())

    errorindex, pos_inblocks = sort_by_kmerlength(RefIndexlist, k)

    index = [RefIndexlist[i] for i in range(len(RefIndexlist)) if i not in errorindex]

    index += pos_inblocks

    

    return index



def Var2Hap(Var, founder, k):

    RefIndex = collapsed_adjacent(Var, 1)

    

    Hap = {} # Hap[refpos][haplotype] = [edges]

    Ns = set()

    #print len(RefIndex)

    for refpos in RefIndex:

        if isinstance(refpos, int):

            Var_D = Var[refpos]

        else:

            s,e = refpos # both end included

            Var_D = {}

            sanchor, eanchor = graph.boundingAnchors(refstrain, s,e)

            Path = graph.getSubPath(sanchor, eanchor, init_strain=set(founder))

            

            for itemlist, strains in Path:

                edge = itemlist[1]

                assert edge.startswith('E') or edge.startswith("S")



                sequence = graph.findsequence(itemlist,countinganchor=True)

                alter_strain = list(strains)[0]

                alts, soffset = coordinate_exchange(graph, s, alter_strain, refstrain, k)

                alte, eoffset = coordinate_exchange(graph, e, alter_strain, refstrain, k)

                haplotype = sequence[soffset:eoffset+1]

                Var_D[edge] = {}

                Var_D[edge]['altoffset'] = [range(soffset,eoffset+1)]

                Var_D[edge]['base'] = haplotype

        H = {}

        for edge, D in Var_D.items():

            haplotype = D['base']

            

            if "N" in haplotype: # exlude all Ns

                Ns.add(edge)

                continue

            

            altpos = D.get('altoffset', "?")

            if altpos == "?":

                H[haplotype] = H.get(haplotype, []) + [(edge, refpos)]

            else:

                H[haplotype] = H.get(haplotype, []) + [(edge, altpos)]

            

        # check 

        no_var = []

        if len(H)>1:

            Hap[refpos] = H

        else:

            no_var.append((H, refpos))

            #print(H, refpos)

    return Hap, Ns, no_var



def get_haplotype_table(graph, Hap):

    Info = {}

    for refpos, Var_D in Hap.items():

        for haplotype, items in Var_D.items():

            for edge, altpos in items:

                strainlist = graph.edges[edge]['strain']

                for strain in strainlist:

                    Info[refpos] = Info.get(refpos, {})

                    Info[refpos][strain] = haplotype

    df = pd.DataFrame(Info)

    df = df.dropna(axis=1) # delete position with ambiguous sequences

    return df



def sortcolumns(data):

    columns = []

    for i in data.columns:

        if isinstance(i, int):

            columns.append(i)

        else:

            columns.append(i[0])



    index = numpy.argsort(columns)

    colorder = [data.columns[i] for i in index]

    data = data.loc[:,colorder]

    return data
# Run

founder = getfounders(graph) # strain list

k = 11 # kmer length

refstrain = 'MN908947'

Var = find_allVar(graph,refstrain, founder, k)

print(len(Var))

Hap, Nedge, no_var = Var2Hap(Var, founder, k)

df = get_haplotype_table(graph, Hap)



data = sortcolumns(df)

data.to_csv('CovVariants.csv')

data.shape
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import numpy as np

import matplotlib

import matplotlib.pyplot as plt



import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from mpl_toolkits.axes_grid1 import make_axes_locatable





def Plot_Indels(df,roworder = df.index, shift = -0.5, excludesample = [], excludepos = []):

    '''Import the pandas dataframe where columns correspond to ref intervals, rows correspond to strains, 

    elements denote haplotypes

     roworder is the order for samples, to get a pretty plot, input the clustered order

     

    '''

    excludesample = [unicode(item) for item in excludesample]

    roworder = [item for item in roworder if item not in excludesample]

    

    df = df.drop(excludesample, axis=0)

    df = df.drop(excludepos, axis=1)

    

    def filternonvariantsposition(df, roworder):

        data = df.loc[roworder]

        colorder = data.columns



        # filter position without variants (according sample set)

        delcolumns = []

        for i in range(len(colorder)):

            if len(set(data.iloc[:,i].values)) < 2:

                delcolumns.append(colorder[i])

        colorder = [item for item in colorder if item not in delcolumns]





        data = data.loc[:,colorder]

        return data



    data = filternonvariantsposition(df, roworder)

    #print "data", data.shape



    def expandforplot(data):

        columns = []

        for i in data.columns:

            if isinstance(i, int):

                columns.append(i)

            else:

                columns.append(i[0])



        index = numpy.argsort(columns)

        colorder = [data.columns[i] for i in index]

        data = data.loc[:,colorder]

        p = data.values            





        def expand_indels(index, p):

            r,c = p.shape

            #print p[:,index]

            m = [len(seq) for seq in p[:,index]]

            m = max(m)

            a = numpy.ndarray([r,m], dtype=object)



            for i, seq in enumerate(p[:,index]):

                for j in range(len(seq)):

                    a[i,m - len(seq) + j] = seq[j]

            return a







        def expandmatrix(p):

            mydata = expand_indels(0, p)

            labels_y = [0] + [""] * (mydata.shape[1]-2) + [33]

            r, c = p.shape

            for index in range(1,c):

                index_ = colorder[index]

                if isinstance(index_, int):

                    mydata = numpy.hstack((mydata, p[:,index].reshape(-1,1)))

                    assert len(set(p[:,index]))>1

                    labels_y += [index_]

                else:

                    a = expand_indels(index, p)

                    mydata = numpy.hstack((mydata, a))

                    labels_y += [index_[0]] + [""] * (a.shape[1]-2) + [index_[1]]

            return mydata, labels_y



        mydata, labels_y = expandmatrix(p)



        return mydata, labels_y



    mydata, labels_y = expandforplot(data)

    #print mydata.shape, labels_y

    

    

    # exchange character to number for plot

    character = set()

    for i in mydata:

        for j in i:

            character.add(j)



    colorD = {}

    count = 1

    for i in ['A', 'C', 'G', 'T', 'S', 'R', 'W', 'Y']:

        colorD[i] = count

        count += 1



    r,c = mydata.shape

    p = mydata

    

    

    for i in range(r):

        for j in range(c):

            p[i,j] = colorD.get(p[i,j], 0)

    p = p.astype(float)

    

    # plot

    colorDict = {'A': 'blue', "G":'green', 'C':'red', 'T': 'yellow', "None": 'grey'}

    cmap = mpl.colors.ListedColormap(["white",'blue', 'red', 'green', 'yellow', 'cyan', 'pink', 'grey','orange'], name = 'colors', N = None)

    n = mpl.colors.Normalize(vmin=0,vmax=3)



    fig = plt.figure(figsize=[20,40])

    AX = plt.gca()

    im = AX.matshow(p, cmap=cmap)

    plt.xlim([shift-0.5,c])

    plt.ylim([shift-0.5,r])

    # set ticks

    plt.yticks(range(0,r), roworder,rotation=0, fontsize = 5)

    plt.xticks(range(0,c), labels_y, rotation = 80, fontsize = 5)



#     #plot grid

    r,c = p.shape

    

    ax = np.arange(shift,c,1)

    for z in ax:

        plt.plot((z,z),(shift ,r+shift),'k-', linewidth=0.2)



    ax = np.arange(shift,r,1)

    for z in ax:

        plt.plot((shift,c+shift),(z,z),'k-', linewidth=0.2)



    character = ["",'A', 'C', 'G', 'T', 'S', 'R', 'W', 'Y']

    divider = make_axes_locatable(AX)

    cax = divider.append_axes("right", size="1%", pad=0.05)



    cbar = plt.colorbar(im, cax=cax)

    cbar.ax.get_yaxis().set_ticks([])



    # annotate characters

    for j, lab in enumerate(character):

        cbar.ax.text(.5, (j), lab, ha='center', va='center',fontsize = 10)

    # cbar.ax.get_yaxis().labelpad = 15

    # cbar.ax.set_ylabel('# of contacts', rotation=270)



    # save

    plt.show()

    fig.savefig("Variantsdisply.png")
roworder = data.index

print( roworder)

Plot_Indels(data,roworder, excludepos = [])
import seaborn as sns



def binary_data(df):

    r,c = df.shape

    values = df.values

    binary = numpy.zeros([r,c])

    for i in range(c):

        hap = values[:,i]

        h, c = numpy.unique(hap, return_counts= True)

        D = {}

        for j in range(len(h)):

            if c[j] == max(c):

                D[h[j]] = 0

            else:

                D[h[j]] = 1



        for x in range(r):

            binary[x,i] = D[values[x,i]]

    return binary



f, ax = plt.subplots(figsize=(11, 9))

binary = binary_data(data)

binary_table = pd.DataFrame(binary, columns= df.columns, index = df.index)

# Generate a custom diverging colormap

#cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.set(font_scale=1.5)

# Draw the heatmap with the mask and correct aspect ratio

sns_plot = sns.heatmap(binary_table)

# , cmap=cmap, vmax=1, center=0,

#                        xticklabels = True, yticklabels=True, annot_kws={"size": 0.5},

#                         square=False, linewidths=.5)

# save to file

fig = sns_plot.get_figure()

fig.savefig("BinaryTable.png")

# get info 

def get_sample_info(table):

    sampleinfo = pd.read_csv(table, header=0, index_col=None)

    def isNaN(num):

        return num != num



    SampleD = {}

    for i in range(len(sampleinfo)):

        sample = sampleinfo['Accession'][i]

        info = str(sampleinfo['Geo_Location'][i])

        SampleD[sample] = info

    return SampleD



table = '../input/graphicalgenomeapi/NCBI_seqinfo.csv'

SampleD = get_sample_info(table)
# Sample Visualization : MDS



from sklearn.manifold import MDS

from sklearn.metrics.pairwise import euclidean_distances

from scipy.cluster import hierarchy

import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform



binary = binary_data(data)



Y= pdist(binary, 'cityblock')

binary_d = squareform(Y)

label = [SampleD[item] for item in df.index]

mds = MDS(n_components=2)

#X_transformed = embedding.fit_transform(binary_d)

transpos = mds.fit(binary_d).embedding_





fig = plt.figure(1)

ax = plt.axes([0., 0., 1., 1.])





plt.scatter(transpos[:, 0], transpos[:, 1], color='red',lw=0, label='MDS')#

for i in range(len(transpos)):

    plt.text(transpos[i, 0], transpos[i, 1], label[i], fontsize=10, alpha = 0.5, c = 'grey', rotation = -45)

plt.legend(scatterpoints=1, loc='best', shadow=False)

fig.savefig('MDScluster.png')
# Sample Subtypes and Identity Mutations



import pylab

import scipy.cluster.hierarchy as sch



fig = pylab.figure(figsize=(10,12))

binary = binary_data(data)

print(binary.shape)

# Compute and plot sample dendrogram.



dist = pdist(binary, 'cityblock')

ax1 = fig.add_axes([0.09,0.1,0.2,0.6])

Y = sch.linkage(dist, method='average', metric = 'cityblock')

Z1 = sch.dendrogram(Y, orientation='left')

ax1.set_xticks([])

ax1.set_yticks([])



# Compute and plot position dendrogram.

dist1 = pdist(binary.T, 'cityblock')

ax2 = fig.add_axes([0.3,0.71,0.6,0.2])

Y2 = sch.linkage(dist1, method='average', metric = 'cityblock')

Z2 = sch.dendrogram(Y2)

ax2.set_xticks([])

ax2.set_yticks([])



# Plot distance matrix

axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])

idx1 = Z1['leaves']

idx2 = Z2['leaves']

b = binary[idx1,:]

b = b[:,idx2]



im = axmatrix.matshow(b, aspect='auto', origin='lower', cmap=pylab.cm.Blues)

axmatrix.set_xticks([])

axmatrix.set_yticks([])



fig.savefig("VariantsCluster.png")
from scipy.cluster.hierarchy import fcluster

samplenames = data.index

fl = fcluster(Y,7,criterion='maxclust')

print(fl[idx1])

#print( numpy.unique(fl, return_counts= True))

for c in [1,2,3]:

    print()

    cluster1 = numpy.where(fl == c)[0]

    clusterinfo = [SampleD[samplenames[item]] for item in cluster1]

    place, occurrence = numpy.unique(clusterinfo,return_counts = True)

    clusterinfo = dict(zip(place, occurrence))

    print("Cluster%s" % str(c), "%s samples" % str(len(cluster1)), clusterinfo)

    # mutations

    mutationpos = set(numpy.where(binary[cluster1,:] == 1)[1])

    

    for pos in mutationpos:

        if sum(binary[cluster1,pos]) > len(cluster1)/4.0:

            r = numpy.where(binary[cluster1,pos]==1)[0]

            print("RefCoord", data.columns[pos], "WithinClusterFrequency:", sum(binary[cluster1,pos])/len(cluster1), 

                  "Rare Allel:", "".join(list(set(data.values[cluster1,pos][r]))))

    
# samples characterized by mutation within S (23,403bp)

index = numpy.where(data[23403] == 'G')[0]

S_data = data[data[23403] == 'G']

S_b = binary[index,:]

S_specific = S_data.index



clusterinfo = [SampleD[item] for item in S_specific]

place, occurrence = numpy.unique(clusterinfo,return_counts = True)

clusterinfo = dict(zip(place, occurrence))

print(len(S_specific), "Spike-mutation(23,403) samples", clusterinfo)



# mutations

mutationpos = set(numpy.where(S_b == 1)[1])

#print(mutationpos)

for pos in mutationpos:

    #print(sum(S_b[:,pos]), S_data.columns[pos])

    if sum(S_b[:,pos]) > len(S_specific)/10.0:

        r = numpy.where(S_b[:,pos]==1)[0]

        print("RefCoord", data.columns[pos], "WithinClusterFrequency:", sum(S_b[:,pos])/len(S_specific), 

              "Rare Allel:", "".join(list(set(S_data.values[:,pos][r]))))

# identity mutations



positionlist = data.columns

for i in Z2['leaves'][:20]:

    p = positionlist[i]

    for s, e, n in interval:

        if isinstance(p, int):

            pos = int(p)

        else:

            pos = p[0]

        if pos >= s and pos <= e:

            break

    hap, count = numpy.unique(data.iloc[:,i], return_counts=True)

    print( p,n, dict(zip(hap, count)))