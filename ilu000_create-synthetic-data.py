import numpy as np

import matplotlib.pyplot as plt
sec = 52   # Number of seconds to produce, delete first two seconds as they are bursts

dt = 1 / 10000  # seconds

# samples needs to be enough to be generating long enough strips

# but it's length is stochastic!

samples = 100000    # 150000 for 50 secs

samplesout = int(sec / dt)

maxchannels = 3

maxrecords = 5  # how many results do we need?

time = np.linspace(0, samples - 1, samples)  # test this!!

# filename = 'states3n.xlsx'

# this scheme is the fast open scheme (used for 1, 3, 5 channels and 10 channels)

# fiona (fast open)

scheme = np.array([[0, 4799, 109, 0, 0, 0],

                   [3368, 0, 0, 0, 353, 0],

                   [262, 0, 0, 24, 0, 0],

                   [0, 0, 638, 0, 0, 0],

                   [0, 218, 0, 0, 0, 11.5],

                   [0, 0, 0, 0, 31, 0]])

isopen = [0, 1, 0, 0, 1, 1]



# this scheme is not evaluated yet (maybe same behavior, maybe slow opening?)

# lowery

scheme2 = np.array([[0, 0.37, 0, 0, 0],

                    [286, 0, 1856, 811, 0],

                    [0, 37, 0, 0, 2395],

                    [0, 148, 0, 0, 0],

                    [0, 0, 5144, 0, 0]])

isopen2 = [0, 0, 0, 1, 1]  # is maybe wrong in the github? (could be [0, 1, 1, 0, 0])



# outcomment if lowery should not be used

# scheme = scheme2

# isopen = isopen2

# samples = samples // 10



maxstates = len(scheme)

state = 1





def getalldists(maxstates, scheme, samples):

    alldist = [0] * (maxstates)

    for lstate in range(maxstates):

        trans = []

        for j in range(-maxstates, maxstates):

            if (0 <= (lstate + j)) and ((lstate + j) < maxstates):

                if scheme[lstate, lstate + j] > 0:

                    # sort out the preallocation for speed

                    row = lstate

                    col = lstate + j

                    val = scheme[row, col]

                    trans.append([lstate, val])

        f = np.random.exponential((1 / np.array(trans)[:, 1].sum()), size=[samples, 1])

        # f(randperm(length(f))); unecessary

        alldist[lstate] = f

    highest = 0

    for ii in range(len(alldist)):

        if alldist[ii].mean() > highest:

            commonest = ii

            highest = alldist[ii].mean()

    # str1=["commonest=",num2str(commonest)];

    # print(str1);

    return alldist, commonest





def neigbours(lstate, maxstates, scheme):

    trans = []

    for j in range(-maxstates, maxstates):

        if (0 <= (lstate + j)) and ((lstate + j) < maxstates):

            if scheme[lstate, lstate + j] > 0:

                # sort out the preallocation for speed

                row = lstate

                col = lstate + j

                val = scheme[row, col]

                trans.append([col, val])

                if trans[0][1] == 0:

                    print("bug down here should never be no possible transitisions")

    # print(trans)

    return np.array(trans)





def nextstate(state, maxstates, scheme):

    # This was my original idea, but a similar one is here:

    # http://www.scholarpedia.org/article/Stochastic_models_of_ion_channel_gating

    trans = neigbours(state, maxstates, scheme)

    # Now MLG from the above site for massive speed improvement.

    if len(trans) == 1:

        state = trans[0, 0]

    else:

        psum = trans[:, 1].sum()

        ran = np.random.rand(1) * psum

        for row in range(len(trans[:, 0])):

            smmer = trans[:row + 1, 1].sum()

            if (smmer >= ran[0]):

                state = trans[row, 0]

                # print(int(state))

                return int(state)

    # print(int(state))

    return int(state)





def getlifetime(nth, dists, state):

    # print(nth);

    thislife = dists[state][nth][0]

    return thislife





def maketimeseries(lifetimes, dt):

    lifetimes = np.array(lifetimes)

    # output = np.zeros([2000000, 2])  # Needs to be more than samples in length

    output = np.array([[0], [0]])

    # cur = 0

    # tlen = 0

    for lifetime in lifetimes:

        thislen = lifetime[0]

        thisstate = lifetime[1]

        num = int(round(thislen / dt))

        # tlen = num + tlen

        t1 = output[0].max() + dt

        t2 = t1 + thislen

        a = np.linspace(t1, t2, num)

        values = (np.ones_like(a) * thisstate).astype(int)

        output = np.concatenate([output, [a, values]], axis=1)

        # output[cur:cur + num, 0] = a

        # output[cur:cur + num, 1] = int(thisstate)

        # cur = cur + num

    # output=output(1:tlen,:)

    return output.T





def fstatesn(scheme, isopen, dt, samples, time, state, maxstates):

    # C-O-O

    # READ FROM A TABLE!!

    # State 1 = colA

    # state 2

    # startstate=1+int64(rand(1,1)*(maxstates-1))

    disters, commonest = getalldists(maxstates, scheme, samples)

    # THESE NEXT LINES MUST CHANGE IF MAXSTATES INCREASE FROM 5

    open_state = np.zeros(maxstates)

    for event in range(maxstates):

        if isopen[event] == 1:

            open_state[event] = event

    open_state = open_state[open_state > 0]

    lifetimes = []

    # states=[];

    for ii in range(samples):

        # look up the correct range of kf and kb on the basis of the current state

        tl = getlifetime(ii, disters, state)

        state = nextstate(state, maxstates, scheme)

        # states=[states;state];

        # print(state);

        if state in open_state:

            lifetimes.append([tl, 1])

        else:

            lifetimes.append([tl, 0])

    out = maketimeseries(lifetimes, dt)

    state = commonest

    return out, state





for record in range(maxrecords):

    print("Record =", record)

    # Multichannels

    out = np.zeros([samplesout, 2])

    channels = 0

    shorty = 0

    while channels < maxchannels:

        channels = channels + 1

        print("Channel =", channels)

        temp, state = fstatesn(scheme, isopen, dt, samples, time, state, maxstates)

        if len(temp) > samplesout:

            shorty = 0  # reset the time saver variable shorty.

            if channels == 1:

                out[:samplesout, :] = temp[:samplesout, :]

            else:

                out[:samplesout, 1] = out[:samplesout, 1] + temp[:samplesout, 1]

            samples = int(0.9 * samples)    # may be wasting time to go with shorter search

            print("too long?", samples)

        else:

            shorty = shorty + 1

            if shorty > 2:

                shorty = 0

                samples = int(1.5 * samples)  # we didn't get enough samples so increase and go again.

                # now do this channel number again ignore warning!

            print("too short", samples)

            channels = channels - 1

    filename = 'astr' + str(record) + '_' + str(maxchannels) + 'c_fast_open.npy'

    np.save(filename, out)



    # Crops of the first 2 seconds because of the bug I cannot trace...

    # that always starts the channel in a burst.

    # csvwrite(['astr' num2str(record) '.csv'],out(:,2));



    plt.plot(out[:, 0], out[:, 1])

    plt.ylim([0, maxchannels + 1])

    plt.show()