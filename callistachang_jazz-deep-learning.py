!pip install music21
from __future__ import print_function



# data science libraries

import numpy as np

import pandas as pd



# machine learning libraries

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM

# import tensorflow as tf



# music libraries

from music21 import *



# utility libraries

from itertools import groupby, zip_longest #zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-

from collections import defaultdict, OrderedDict

import random

import copy

import pdb

import sys

import os



# device_name = tf.test.gpu_device_name()

# if "GPU" not in device_name:

#     print("GPU device not found")

# print('Found GPU at: {}'.format(device_name))
#----------------------------HELPER FUNCTIONS----------------------------------#



''' Helper function to down num to the nearest multiple of mult. '''

def __roundDown(num, mult):

    return (float(num) - (float(num) % mult))



''' Helper function to round up num to nearest multiple of mult. '''

def __roundUp(num, mult):

    return __roundDown(num, mult) + mult



''' Helper function that, based on if upDown < 0 or upDown >= 0, rounds number 

    down or up respectively to nearest multiple of mult. '''

def __roundUpDown(num, mult, upDown):

    if upDown < 0:

        return __roundDown(num, mult)

    else:

        return __roundUp(num, mult)



''' Helper function, from recipes, to iterate over list in chunks of n 

    length. '''

def __grouper(iterable, n, fillvalue=None):

    args = [iter(iterable)] * n

    return zip_longest(*args, fillvalue=fillvalue)
#----------------------------PUBLIC FUNCTIONS----------------------------------#



''' Smooth the measure, ensuring that everything is in standard note lengths 

    (e.g., 0.125, 0.250, 0.333 ... ). '''

def prune_grammar(curr_grammar):

    pruned_grammar = curr_grammar.split(' ')



    for ix, gram in enumerate(pruned_grammar):

        terms = gram.split(',')

        terms[1] = str(__roundUpDown(float(terms[1]), 0.250, 

            random.choice([-1, 1])))

        pruned_grammar[ix] = ','.join(terms)

    pruned_grammar = ' '.join(pruned_grammar)



    return pruned_grammar



''' Remove repeated notes, and notes that are too close together. '''

def prune_notes(curr_notes):

    for n1, n2 in __grouper(curr_notes, n=2):

        if n2 == None: # corner case: odd-length list

            continue

        if isinstance(n1, note.Note) and isinstance(n2, note.Note):

            if n1.nameWithOctave == n2.nameWithOctave:

                curr_notes.remove(n2)



    return curr_notes



''' Perform quality assurance on notes. '''

def clean_up_notes(curr_notes):

    removeIxs = []

    for ix, m in enumerate(curr_notes):

        # QA1: ensure nothing is of 0 quarter note len, if so changes its len

        if (m.quarterLength == 0.0):

            m.quarterLength = 0.250

        # QA2: ensure no two melody notes have same offset, i.e. form a chord.

        # Sorted, so same offset would be consecutive notes.

        if (ix < (len(curr_notes) - 1)):

            if (m.offset == curr_notes[ix + 1].offset and

                isinstance(curr_notes[ix + 1], note.Note)):

                removeIxs.append((ix + 1))

    curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]



    return curr_notes
''' Build a 2-layer LSTM from a training corpus. '''

def build_model(corpus, val_indices, max_len, N_epochs=128):

    # number of different values or words in corpus

    N_values = len(set(corpus))



    # cut the corpus into semi-redundant sequences of max_len values

    step = 3

    sentences = []

    next_values = []

    for i in range(0, len(corpus) - max_len, step):

        sentences.append(corpus[i: i + max_len])

        next_values.append(corpus[i + max_len])

    print('[@] Number of sequences:', len(sentences))



    # transform data into binary matrices

    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)

    y = np.zeros((len(sentences), N_values), dtype=np.bool)

    for i, sentence in enumerate(sentences):

        for t, val in enumerate(sentence):

            X[i, t, val_indices[val]] = 1

        y[i, val_indices[next_values[i]]] = 1



    # build a 2 stacked LSTM

    model = Sequential()

    model.add(LSTM(128, return_sequences=True, input_shape=(max_len, N_values)))

    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(N_values))

    model.add(Activation('softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    

    model.fit(X, y, batch_size=128, epochs=N_epochs, verbose=1)



    return model
#----------------------------HELPER FUNCTIONS----------------------------------#



''' Helper function to determine if a note is a scale tone. '''

def __is_scale_tone(chord, note):

    # Method: generate all scales that have the chord notes th check if note is

    # in names



    # Derive major or minor scales (minor if 'other') based on the quality

    # of the chord.

    scaleType = scale.DorianScale() # i.e. minor pentatonic

    if chord.quality == 'major':

        scaleType = scale.MajorScale()

    # Can change later to deriveAll() for flexibility. If so then use list

    # comprehension of form [x for a in b for x in a].

    scales = scaleType.derive(chord) # use deriveAll() later for flexibility

    allPitches = list(set([pitch for pitch in scales.getPitches()]))

    allNoteNames = [i.name for i in allPitches] # octaves don't matter



    # Get note name. Return true if in the list of note names.

    noteName = note.name

    return (noteName in allNoteNames)



''' Helper function to determine if a note is an approach tone. '''

def __is_approach_tone(chord, note):

    # Method: see if note is +/- 1 a chord tone.



    for chordPitch in chord.pitches:

        stepUp = chordPitch.transpose(1)

        stepDown = chordPitch.transpose(-1)

        if (note.name == stepDown.name or 

            note.name == stepDown.getEnharmonic().name or

            note.name == stepUp.name or

            note.name == stepUp.getEnharmonic().name):

                return True

    return False



''' Helper function to determine if a note is a chord tone. '''

def __is_chord_tone(lastChord, note):

    return (note.name in (p.name for p in lastChord.pitches))



''' Helper function to generate a chord tone. '''

def __generate_chord_tone(lastChord):

    lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]

    return note.Note(random.choice(lastChordNoteNames))



''' Helper function to generate a scale tone. '''

def __generate_scale_tone(lastChord):

    # Derive major or minor scales (minor if 'other') based on the quality

    # of the lastChord.

    scaleType = scale.WeightedHexatonicBlues() # minor pentatonic

    if lastChord.quality == 'major':

        scaleType = scale.MajorScale()

    # Can change later to deriveAll() for flexibility. If so then use list

    # comprehension of form [x for a in b for x in a].

    scales = scaleType.derive(lastChord) # use deriveAll() later for flexibility

    allPitches = list(set([pitch for pitch in scales.getPitches()]))

    allNoteNames = [i.name for i in allPitches] # octaves don't matter



    # Return a note (no octave here) in a scale that matches the lastChord.

    sNoteName = random.choice(allNoteNames)

    lastChordSort = lastChord.sortAscending()

    sNoteOctave = random.choice([i.octave for i in lastChordSort.pitches])

    sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))

    return sNote



''' Helper function to generate an approach tone. '''

def __generate_approach_tone(lastChord):

    sNote = __generate_scale_tone(lastChord)

    aNote = sNote.transpose(random.choice([1, -1]))

    return aNote



''' Helper function to generate a random tone. '''

def __generate_arbitrary_tone(lastChord):

    return __generate_scale_tone(lastChord) # fix later, make random note.
#----------------------------PUBLIC FUNCTIONS----------------------------------#



''' Given the notes in a measure ('measure') and the chords in that measure

    ('chords'), generate a list of abstract grammatical symbols to represent 

    that measure as described in GTK's "Learning Jazz Grammars" (2009). 

    Inputs: 

    1) "measure" : a stream.Voice object where each element is a

        note.Note or note.Rest object.

        >>> m1

        <music21.stream.Voice 328482572>

        >>> m1[0]

        <music21.note.Rest rest>

        >>> m1[1]

        <music21.note.Note C>

        Can have instruments and other elements, removes them here.

    2) "chords" : a stream.Voice object where each element is a chord.Chord.

        >>> c1

        <music21.stream.Voice 328497548>

        >>> c1[0]

        <music21.chord.Chord E-4 G4 C4 B-3 G#2>

        >>> c1[1]

        <music21.chord.Chord B-3 F4 D4 A3>

        Can have instruments and other elements, removes them here. 

    Outputs:

    1) "fullGrammar" : a string that holds the abstract grammar for measure.

        Format: 

        (Remember, these are DURATIONS not offsets!)

        "R,0.125" : a rest element of  (1/32) length, or 1/8 quarter note. 

        "C,0.125<M-2,m-6>" : chord note of (1/32) length, generated

                             anywhere from minor 6th down to major 2nd down.

                             (interval <a,b> is not ordered). '''

def parse_melody(fullMeasureNotes, fullMeasureChords):

    # Remove extraneous elements.x

    measure = copy.deepcopy(fullMeasureNotes)

    chords = copy.deepcopy(fullMeasureChords)

    measure.removeByNotOfClass([note.Note, note.Rest])

    chords.removeByNotOfClass([chord.Chord])



    # Information for the start of the measure.

    # 1) measureStartTime: the offset for measure's start, e.g. 476.0.

    # 2) measureStartOffset: how long from the measure start to the first element.

    measureStartTime = measure[0].offset - (measure[0].offset % 4)

    measureStartOffset  = measure[0].offset - measureStartTime



    # Iterate over the notes and rests in measure, finding the grammar for each

    # note in the measure and adding an abstract grammatical string for it. 



    fullGrammar = ""

    prevNote = None # Store previous note. Need for interval.

    numNonRests = 0 # Number of non-rest elements. Need for updating prevNote.

    for ix, nr in enumerate(measure):

        # Get the last chord. If no last chord, then (assuming chords is of length

        # >0) shift first chord in chords to the beginning of the measure.

        try: 

            lastChord = [n for n in chords if n.offset <= nr.offset][-1]

        except IndexError:

            chords[0].offset = measureStartTime

            lastChord = [n for n in chords if n.offset <= nr.offset][-1]



        # FIRST, get type of note, e.g. R for Rest, C for Chord, etc.

        # Dealing with solo notes here. If unexpected chord: still call 'C'.

        elementType = ' '

        # R: First, check if it's a rest. Clearly a rest --> only one possibility.

        if isinstance(nr, note.Rest):

            elementType = 'R'

        # C: Next, check to see if note pitch is in the last chord.

        elif nr.name in lastChord.pitchNames or isinstance(nr, chord.Chord):

            elementType = 'C'

        # L: (Complement tone) Skip this for now.

        # S: Check if it's a scale tone.

        elif __is_scale_tone(lastChord, nr):

            elementType = 'S'

        # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.

        elif __is_approach_tone(lastChord, nr):

            elementType = 'A'

        # X: Otherwise, it's an arbitrary tone. Generate random note.

        else:

            elementType = 'X'



        # SECOND, get the length for each element. e.g. 8th note = R8, but

        # to simplify things you'll use the direct num, e.g. R,0.125

        if (ix == (len(measure)-1)):

            # formula for a in "a - b": start of measure (e.g. 476) + 4

            diff = measureStartTime + 4.0 - nr.offset

        else:

            diff = measure[ix + 1].offset - nr.offset



        # Combine into the note info.

        noteInfo = "%s,%.3f" % (elementType, nr.quarterLength) # back to diff



        # THIRD, get the deltas (max range up, max range down) based on where

        # the previous note was, +- minor 3. Skip rests (don't affect deltas).

        intervalInfo = ""

        if isinstance(nr, note.Note):

            numNonRests += 1

            if numNonRests == 1:

                prevNote = nr

            else:

                noteDist = interval.Interval(noteStart=prevNote, noteEnd=nr)

                noteDistUpper = interval.add([noteDist, "m3"])

                noteDistLower = interval.subtract([noteDist, "m3"])

                intervalInfo = ",<%s,%s>" % (noteDistUpper.directedName, 

                    noteDistLower.directedName)

                # print "Upper, lower: %s, %s" % (noteDistUpper,

                #     noteDistLower)

                # print "Upper, lower dnames: %s, %s" % (

                #     noteDistUpper.directedName,

                #     noteDistLower.directedName)

                # print "The interval: %s" % (intervalInfo)

                prevNote = nr



        # Return. Do lazy evaluation for real-time performance.

        grammarTerm = noteInfo + intervalInfo 

        fullGrammar += (grammarTerm + " ")



    return fullGrammar.rstrip()



''' Given a grammar string and chords for a measure, returns measure notes. '''

def unparse_grammar(m1_grammar, m1_chords):

    m1_elements = stream.Voice()

    currOffset = 0.0 # for recalculate last chord.

    prevElement = None

    for ix, grammarElement in enumerate(m1_grammar.split(' ')):

        terms = grammarElement.split(',')

        currOffset += float(terms[1]) # works just fine



        # Case 1: it's a rest. Just append

        if terms[0] == 'R':

            rNote = note.Rest(quarterLength = float(terms[1]))

            m1_elements.insert(currOffset, rNote)

            continue



        # Get the last chord first so you can find chord note, scale note, etc.

        try: 

            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]

        except IndexError:

            m1_chords[0].offset = 0.0

            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]



        # Case: no < > (should just be the first note) so generate from range

        # of lowest chord note to highest chord note (if not a chord note, else

        # just generate one of the actual chord notes). 



        # Case #1: if no < > to indicate next note range. Usually this lack of < >

        # is for the first note (no precedent), or for rests.

        if (len(terms) == 2): # Case 1: if no < >.

            insertNote = note.Note() # default is C



            # Case C: chord note.

            if terms[0] == 'C':

                insertNote = __generate_chord_tone(lastChord)



            # Case S: scale note.

            elif terms[0] == 'S':

                insertNote = __generate_scale_tone(lastChord)



            # Case A: approach note.

            # Handle both A and X notes here for now.

            else:

                insertNote = __generate_approach_tone(lastChord)



            # Update the stream of generated notes

            insertNote.quarterLength = float(terms[1])

            if insertNote.octave < 4:

                insertNote.octave = 4

            m1_elements.insert(currOffset, insertNote)

            prevElement = insertNote



        # Case #2: if < > for the increment. Usually for notes after the first one.

        else:

            # Get lower, upper intervals and notes.

            interval1 = interval.Interval(terms[2].replace("<",''))

            interval2 = interval.Interval(terms[3].replace(">",''))

            if interval1.cents > interval2.cents:

                upperInterval, lowerInterval = interval1, interval2

            else:

                upperInterval, lowerInterval = interval2, interval1

            lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)

            highPitch = interval.transposePitch(prevElement.pitch, upperInterval)

            numNotes = int(highPitch.ps - lowPitch.ps + 1) # for range(s, e)



            # Case C: chord note, must be within increment (terms[2]).

            # First, transpose note with lowerInterval to get note that is

            # the lower bound. Then iterate over, and find valid notes. Then

            # choose randomly from those.

            

            if terms[0] == 'C':

                relevantChordTones = []

                for i in range(numNotes):

                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())

                    if __is_chord_tone(lastChord, currNote):

                        relevantChordTones.append(currNote)

                if len(relevantChordTones) > 1:

                    insertNote = random.choice([i for i in relevantChordTones

                        if i.nameWithOctave != prevElement.nameWithOctave])

                elif len(relevantChordTones) == 1:

                    insertNote = relevantChordTones[0]

                else: # if no choices, set to prev element +-1 whole step

                    insertNote = prevElement.transpose(random.choice([-2,2]))

                if insertNote.octave < 3:

                    insertNote.octave = 3

                insertNote.quarterLength = float(terms[1])

                m1_elements.insert(currOffset, insertNote)



            # Case S: scale note, must be within increment.

            elif terms[0] == 'S':

                relevantScaleTones = []

                for i in range(numNotes):

                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())

                    if __is_scale_tone(lastChord, currNote):

                        relevantScaleTones.append(currNote)

                if len(relevantScaleTones) > 1:

                    insertNote = random.choice([i for i in relevantScaleTones

                        if i.nameWithOctave != prevElement.nameWithOctave])

                elif len(relevantScaleTones) == 1:

                    insertNote = relevantScaleTones[0]

                else: # if no choices, set to prev element +-1 whole step

                    insertNote = prevElement.transpose(random.choice([-2,2]))

                if insertNote.octave < 3:

                    insertNote.octave = 3

                insertNote.quarterLength = float(terms[1])

                m1_elements.insert(currOffset, insertNote)



            # Case A: approach tone, must be within increment.

            # For now: handle both A and X cases.

            else:

                relevantApproachTones = []

                for i in range(numNotes):

                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())

                    if __is_approach_tone(lastChord, currNote):

                        relevantApproachTones.append(currNote)

                if len(relevantApproachTones) > 1:

                    insertNote = random.choice([i for i in relevantApproachTones

                        if i.nameWithOctave != prevElement.nameWithOctave])

                elif len(relevantApproachTones) == 1:

                    insertNote = relevantApproachTones[0]

                else: # if no choices, set to prev element +-1 whole step

                    insertNote = prevElement.transpose(random.choice([-2,2]))

                if insertNote.octave < 3:

                    insertNote.octave = 3

                insertNote.quarterLength = float(terms[1])

                m1_elements.insert(currOffset, insertNote)



            # update the previous element.

            prevElement = insertNote



    return m1_elements    
#----------------------------HELPER FUNCTIONS----------------------------------#



''' Helper function to parse a MIDI file into its measures and chords '''

def __parse_midi(data_fn):

    # Parse the MIDI data for separate melody and accompaniment parts.

    midi_data = converter.parse(data_fn)

    # Get melody part, compress into single voice.

    melody_stream = midi_data[5]     # For Metheny piece, Melody is Part #5.

    melody1, melody2 = melody_stream.getElementsByClass(stream.Voice)

    for j in melody2:

        melody1.insert(j.offset, j)

    melody_voice = melody1



    for i in melody_voice:

        if i.quarterLength == 0.0:

            i.quarterLength = 0.25



    # Change key signature to adhere to comp_stream (1 sharp, mode = major).

    # Also add Electric Guitar. 

    melody_voice.insert(0, instrument.ElectricGuitar())

    melody_voice.insert(0, key.KeySignature(sharps=1))



    # The accompaniment parts. Take only the best subset of parts from

    # the original data. Maybe add more parts, hand-add valid instruments.

    # Should add least add a string part (for sparse solos).

    # Verified are good parts: 0, 1, 6, 7 '''

    partIndices = [0, 1, 6, 7]

    comp_stream = stream.Voice()

    comp_stream.append([j.flat for i, j in enumerate(midi_data) 

        if i in partIndices])



    # Full stream containing both the melody and the accompaniment. 

    # All parts are flattened. 

    full_stream = stream.Voice()

    for i in range(len(comp_stream)):

        full_stream.append(comp_stream[i])

    full_stream.append(melody_voice)



    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).

    # Note that for different instruments (with stream.flat), you NEED to use

    # stream.Part(), not stream.Voice().

    # Accompanied solo is in range [478, 548)

    solo_stream = stream.Voice()

    for part in full_stream:

        curr_part = stream.Part()

        curr_part.append(part.getElementsByClass(instrument.Instrument))

        curr_part.append(part.getElementsByClass(tempo.MetronomeMark))

        curr_part.append(part.getElementsByClass(key.KeySignature))

        curr_part.append(part.getElementsByClass(meter.TimeSignature))

        curr_part.append(part.getElementsByOffset(476, 548, 

                                                  includeEndBoundary=True))

        cp = curr_part.flat

        solo_stream.insert(cp)



    # Group by measure so you can classify. 

    # Note that measure 0 is for the time signature, metronome, etc. which have

    # an offset of 0.0.

    melody_stream = solo_stream[-1]

    measures = OrderedDict()

    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]

    measureNum = 0 # for now, don't use real m. nums (119, 120)

    for key_x, group in groupby(offsetTuples, lambda x: x[0]):

        measures[measureNum] = [n[1] for n in group]

        measureNum += 1



    # Get the stream of chords.

    # offsetTuples_chords: group chords by measure number.

    chordStream = solo_stream[0]

    chordStream.removeByClass(note.Rest)

    chordStream.removeByClass(note.Note)

    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]



    # Generate the chord structure. Use just track 1 (piano) since it is

    # the only instrument that has chords. 

    # Group into 4s, just like before. 

    chords = OrderedDict()

    measureNum = 0

    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):

        chords[measureNum] = [n[1] for n in group]

        measureNum += 1



    # Fix for the below problem.

    #   1) Find out why len(measures) != len(chords).

    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't

    #           actually show up, while the accompaniment's beat 1 right after does.

    #           Actually on second thought: melody/comp start on Ab, and resolve to

    #           the same key (Ab) so could actually just cut out last measure to loop.

    #           Decided: just cut out the last measure. 

    del chords[len(chords) - 1]

    assert len(chords) == len(measures)



    return measures, chords



''' Helper function to get the grammatical data from given musical data. '''

def __get_abstract_grammars(measures, chords):

    # extract grammars

    abstract_grammars = []

    for ix in range(1, len(measures)):

        m = stream.Voice()

        for i in measures[ix]:

            m.insert(i.offset, i)

        c = stream.Voice()

        for j in chords[ix]:

            c.insert(j.offset, j)

        parsed = parse_melody(m, c)

        abstract_grammars.append(parsed)



    return abstract_grammars
#----------------------------PUBLIC FUNCTIONS----------------------------------#



''' Get musical data from a MIDI file '''

def get_musical_data(data_fn):

    measures, chords = __parse_midi(data_fn)

    abstract_grammars = __get_abstract_grammars(measures, chords)



    return chords, abstract_grammars



''' Get corpus data from grammatical data '''

def get_corpus_data(abstract_grammars):

    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]

    values = set(corpus)

    val_indices = dict((v, i) for i, v in enumerate(values))

    indices_val = dict((i, v) for i, v in enumerate(values))



    return corpus, values, val_indices, indices_val
#----------------------------HELPER FUNCTIONS----------------------------------#



''' Helper function to sample an index from a probability array '''

def __sample(a, temperature=1.0):

    a = np.log(a) / temperature

    dist = np.exp(a) / np.sum(np.exp(a))

    choices = range(len(a))

    return np.random.choice(choices, p=dist)



''' Helper function to generate a predicted value from a given matrix '''

def __predict(model, x, indices_val, diversity):

    preds = model.predict(x, verbose=0)[0]

    next_index = __sample(preds, diversity)

    next_val = indices_val[next_index]



    return next_val



''' Helper function which uses the given model to generate a grammar sequence 

    from a given corpus, indices_val (mapping), abstract_grammars (list), 

    and diversity floating point value. '''

def __generate_grammar(model, corpus, abstract_grammars, values, val_indices,

                       indices_val, max_len, max_tries, diversity):

    curr_grammar = ''

    # np.random.randint is exclusive to high

    start_index = np.random.randint(0, len(corpus) - max_len)

    sentence = corpus[start_index: start_index + max_len]    # seed

    running_length = 0.0

    while running_length <= 4.1:    # arbitrary, from avg in input file

        # transform sentence (previous sequence) to matrix

        x = np.zeros((1, max_len, len(values)))

        for t, val in enumerate(sentence):

            if (not val in val_indices): print(val)

            x[0, t, val_indices[val]] = 1.



        next_val = __predict(model, x, indices_val, diversity)



        # fix first note: must not have < > and not be a rest

        if (running_length < 0.00001):

            tries = 0

            while (next_val.split(',')[0] == 'R' or 

                len(next_val.split(',')) != 2):

                # give up after 1000 tries; random from input's first notes

                if tries >= max_tries:

                    print('Gave up on first note generation after', max_tries, 

                        'tries')

                    # np.random is exclusive to high

                    rand = np.random.randint(0, len(abstract_grammars))

                    next_val = abstract_grammars[rand].split(' ')[0]

                else:

                    next_val = __predict(model, x, indices_val, diversity)



                tries += 1



        # shift sentence over with new value

        sentence = sentence[1:] 

        sentence.append(next_val)



        # except for first case, add a ' ' separator

        if (running_length > 0.00001): curr_grammar += ' '

        curr_grammar += next_val



        length = float(next_val.split(',')[1])

        running_length += length



    return curr_grammar
#----------------------------PUBLIC FUNCTIONS----------------------------------#



''' Generates musical sequence based on the given data filename and settings.

    Plays then stores (MIDI file) the generated output. '''

def generate(data_fn, out_fn, N_epochs):

    # model settings

    max_len = 20

    max_tries = 1000

    diversity = 0.5



    # musical settings

    bpm = 130



    # get data

    print("Getting data...")

    chords, abstract_grammars = get_musical_data(data_fn)

    corpus, values, val_indices, indices_val = get_corpus_data(abstract_grammars)

    print('[@] Corpus length:', len(corpus))

    print('[@] Total # of values:', len(values))



    # build model

    print("Building LSTM model...")

    model = build_model(corpus=corpus, val_indices=val_indices, 

                        max_len=max_len, N_epochs=N_epochs)



    # set up audio stream

    out_stream = stream.Stream()



    # generation loop

    curr_offset = 0.0

    loopEnd = len(chords)

    print(f"Starting generation loop ({loopEnd-1} times)...")

    for loopIndex in range(1, loopEnd):

        # get chords from file

        curr_chords = stream.Voice()

        for j in chords[loopIndex]:

            curr_chords.insert((j.offset % 4), j)



        # generate grammar

        curr_grammar = __generate_grammar(model=model, corpus=corpus, 

                                          abstract_grammars=abstract_grammars, 

                                          values=values, val_indices=val_indices, 

                                          indices_val=indices_val, 

                                          max_len=max_len, max_tries=max_tries,

                                          diversity=diversity)



        curr_grammar = curr_grammar.replace(' A',' C').replace(' X',' C')



        # Pruning #1: smoothing measure

        curr_grammar = prune_grammar(curr_grammar)



        # Get notes from grammar and chords

        curr_notes = unparse_grammar(curr_grammar, curr_chords)



        # Pruning #2: removing repeated and too close together notes

        curr_notes = prune_notes(curr_notes)



        # quality assurance: clean up notes

        curr_notes = clean_up_notes(curr_notes)



        # print # of notes in curr_notes

        print(f'[{loopIndex}] After pruning: {len([i for i in curr_notes if isinstance(i, note.Note)])} notes')



        # insert into the output stream

        for m in curr_notes:

            out_stream.insert(curr_offset + m.offset, m)

        for mc in curr_chords:

            out_stream.insert(curr_offset + mc.offset, mc)



        curr_offset += 4.0



    out_stream.insert(0.0, tempo.MetronomeMark(number=bpm))



    # save stream

    mf = midi.translate.streamToMidiFile(out_stream)

    mf.open(out_fn, 'wb')

    mf.write()

    mf.close()

        

        

''' Runs generate() -- generating, playing, then storing a musical sequence --

    with the default Metheny file. '''

def main(mid_file, n_epochs=128):

    # i/o settings

    data_fn = f'../input/original-metheny/{mid_file}.mid'

    out_fn = f'./generated_{mid_file}_{n_epochs}'

    

    if (n_epochs == 1): 

        out_fn += '_epoch.midi'

    else:               

        out_fn += '_epochs.midi'



    generate(data_fn, out_fn, n_epochs)
main(mid_file='original_metheny', n_epochs=500) # And Then I Knew by Pat Metheny