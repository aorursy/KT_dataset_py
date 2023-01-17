import os, random, time, pickle, argparse, sys, datetime



def load_csv(filename):

    asm = []

    with open(filename) as stream:

        for i, line in enumerate(stream.readlines()):

            if i == 0: continue

            f = int(line.split(',')[-1])

            asm.append(f)

    return asm





def save_csv(asm, filename):

    with open(filename, "w") as s:

        s.write("family_id,assigned_day\n")

        for f, d in enumerate(asm):

            s.write("%d,%d\n" %(f,d+1))



            

class Submission:

    def __init__(self):

        DATAPATH = '../input/santa777/'

        with open (DATAPATH+'Npp.pkl', 'rb') as fp: self.Npp = pickle.load(fp)

        with open (DATAPATH+'Acc_Table.pkl', 'rb') as fp: self.Acc_Table = pickle.load(fp)

        with open (DATAPATH+'Pre_Table.pkl', 'rb') as fp: self.Pre_Table = pickle.load(fp)

        with open (DATAPATH+'Choice_Table.pkl', 'rb') as fp: self.Choice_Table = pickle.load(fp)

        with open (DATAPATH+'Rank_Table.pkl', 'rb') as fp: self.Rank_Table = pickle.load(fp)

        

        self.score = 0

        self.found_a_move = False

        

        self.occ = [0]*100 # current occupancy for 100 days

        self.asm = None # current assignment of days for 5000 families

        self.rank = [0]*5000 # current choice preference for 5000 families

        self.pre = [0.]*5000 # current preference cost for 5000 families

        self.acc = [0.]*100 # current accounting cost for 100 days

        self.moves = []

        self.depth = 0

        self.cnt_moves = 0

        self.total_time = 0

        



    def refresh_mode(self, mode=100, w_pre=1):

        self.mode = mode

        self.w_pre = w_pre

        # Preference Cost

        for f, d in enumerate(self.asm): self.pre[f] = self.Pre_Table[f][d]

        # Accounting Cost

        self.acc[99] = (self.occ[99]-125.0) / 400.0 * self.occ[99]**(0.5)

        for d in range(99): self.acc[d] = self.get_acc_cost(d)

        self.score = ( sum(self.w_pre * self.pre) + sum(self.acc)) / (self.w_pre + 1) * 2

                        

    def shuffle_search(self, n='random', toleration=0, lower_tol=-9999):

        self.found_a_move = False

        k = n if n != 'random' else random.choice([1,2,3,4,5])

        families = random.sample(range(5000), k)

        cands = [random.choice([d for d in self.Choice_Table[f][0:4]  

                                   if d!=self.rank[f]]) for f in families]

        moves = [[families[j], cands[j]] for j in range(k)]

        delta = self.implement_moves(moves, toleration=toleration, lower_tol=lower_tol)

        return delta

        



    def implement_moves(self, moves, toleration=0, lower_tol=-9999):

        _occ = self.occ.copy()

        _asm = self.asm.copy()

        _families, _days = [], []

        for move in moves:

            f, d, n = move[0], move[1], self.Npp[move[0]]

            _occ[ _asm[f] ] -= n

            _occ[ d ] += n

            _families.append(f)

            _days.append( _asm[f] )

            _days.append(d)

            _asm[f] = d



        delta, affected_days = self.compute_delta(_asm, _occ, _families, _days)



        if delta < toleration and delta > lower_tol:

            if ((toleration > 0) and (delta>0)) or (toleration==0): 

                self.found_a_move = True

                self.score += delta 

                self.asm = _asm

                self.occ = _occ

                for f in _families: self.pre[f] = self.Pre_Table[f][_asm[f]]

                for d in affected_days: self.acc[d] = self.get_acc_cost(d, occ_arr=_occ)

                print('Delta', round(delta,3), '\tScore', round(self.score, 4),  

                      '\tmax_occ', max(self.occ), '\tmin_occ', min(self.occ), '\tMode', self.mode)

        return delta





    def compute_delta(self, asm, occ, families, days):

        a = datetime.datetime.now()

        affected_days = days.copy()

        affected_days += [d+1 for d in days if d!=99]

        affected_days += [d-1 for d in days if d!=0]

        days = list(set(affected_days)) # get unique

        

        delta_pre = 0

        for f in families: 

            d = asm[f]

            delta_pre = delta_pre + self.Pre_Table[f][d] - self.pre[f]



        delta_acc = 0

        for d in days: delta_acc = (delta_acc + self.get_acc_cost(d, occ_arr=occ) - self.acc[d])

                

        delta = self.w_pre * delta_pre + delta_acc 

        b = datetime.datetime.now()

        c = b - a

        self.cnt_moves += 1

        self.total_time += c.microseconds

        return delta, affected_days

         

    def embed_submission(self, submission_path):

        self.asm = load_csv(submission_path)

        self.asm = [i-1 for i in self.asm]

        for f, d in enumerate(self.asm): 

            self.occ[d] += self.Npp[f]

            self.rank[f] = int(self.Rank_Table[f][d])

            

    def get_family_info(self, f): 

        return self.asm[f], self.rank[f], self.Npp[f] # day assigned, choice rank, n_people

    

    def get_acc_cost(self, today, occ_arr=None):

        if occ_arr is None: occ_arr = self.occ.copy()

        if today==99: return (occ_arr[99]-125.0) / 400.0 * (occ_arr[99])**(0.5)

        count_violation = 0

        idx0 = occ_arr[today]

        if idx0 < 125: 

            count_violation += abs(idx0-125)

            idx0 = 125

        if idx0 > 300: 

            count_violation += abs(idx0-300)

            idx0 = 300



        idx1 = occ_arr[today+1]

        if idx1 < 125: 

            count_violation += abs(idx1-125)

            idx1 = 125

        if idx1 > 300: 

            count_violation += abs(idx0-300)

            idx1 = 300            

            

        cost = self.Acc_Table[idx0-125][idx1-125] + self.mode*count_violation

        return cost

                



filename = '../input/santa777/submission_76177.csv'

S = Submission()

S.embed_submission(filename)

S.refresh_mode(mode=100)

print('Input score:', S.score)





while S.score < 77777.77:

    delta = S.shuffle_search(toleration=10)

    

    

while S.score > 77777.77:

    delta = S.shuffle_search(lower_tol=-2)



    

upper_tol = 0.5

lower_tol = 0    

while True:

    delta = S.shuffle_search(toleration=upper_tol, lower_tol=lower_tol)

    if S.score > 77777.77:

        upper_tol = 0

        lower_tol = -0.5

    elif S.score < 77777.77:

        upper_tol = 0.5

        lower_tol = 0

    if S.score >= 77777.77 and S.score <= 77777.7749: break

    



print('Done! Score', S.score)

save_csv(S.asm, str(int(S.score))+'.csv')