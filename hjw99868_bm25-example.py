import math
def read_file(file):

    fp = open(file)

    output = dict()

    for line in fp.readlines():

        id, text = line.strip().split('\t')

        output[id] = text

    return output





def example_list2dict(input):

    output = dict()

    for word in input.split():

        if output.get(word) is None:

            output[word] = 0

        output[word] += 1

    return output





def cal_idf(doc_dict):

    doc_num = len(doc_dict)

    idf = dict()

    for doc_id in doc_dict:

        doc_text = list(set(doc_dict[doc_id].split()))

        for word in doc_text:

            if idf.get(word) is None:

                idf[word] = 0

            idf[word] += 1

    for word in idf:

        idf[word] = math.log((doc_num - idf[word] + 0.5) / (idf[word] + 0.5))

    return idf





def bm25(query, doc, idf, avg_doc_len=374):

    k1 = 2.0

    k2 = 1.0

    b = 0.75

    score = 0.0

    for word in query:

        if doc.get(word) == None:

            continue

        W_i = idf[word]

        f_i = doc[word]

        qf_i = query[word]

        doc_len = sum(doc.values())

        K = k1 * (1 - b + b * doc_len / avg_doc_len)

        R1 = f_i * (k1 + 1) / (f_i + K)

        R2 = qf_i * (k2 + 1) / (qf_i + k2)

        R = R1 * R2

        score += W_i * R

    return score
query_dict = read_file('../input/query.txt')

doc_dict = read_file('../input/doc.txt')

idf = cal_idf(doc_dict=doc_dict)

query = example_list2dict(query_dict['0'])

doc = example_list2dict(doc_dict['0_2'])

score = bm25(query, doc, idf)

print('calculate bm25 score between query0 and document0_2: %.4f' % (score))