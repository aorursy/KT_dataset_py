#@title Download MIDI.py and a test Dataset
#MIDI Library
!curl -L "https://github.com/asigalov61/MIDI-TXT-MIDI/raw/master/MIDI.py" > 'MIDI.py'

## MIDI format Mozart Data
!wget http://www.piano-midi.de/zip/mozart.zip
!sudo apt-get install unzip
!unzip mozart.zip -d Dataset
#@title Process MIDI to TXT
encoding_type = "opus-one-byte-encoding" #@param ["score-one-byte-encoding", "opus-one-byte-encoding", "opus-complete-words-encoding"]
enable_sampling = False #@param {type:"boolean"}
sample_length_in_MIDI_events = 2195 #@param {type:"slider", min:0, max:10000, step:1}
advanced_events = True #@param {type:"boolean"}
allow_tempo_changes = True #@param {type:"boolean"}
allow_control_change = True #@param {type:"boolean"}
karaoke = False #@param {type:"boolean"}
debug = False #@param {type:"boolean"}

%cd /content/

# MIDI Dataset to txt dataset converter 
import MIDI
import os
import numpy as np
import tqdm.auto

if os.path.exists("Dataset.txt"):
  os.remove("Dataset.txt")
  print('Removing old Dataset...')
else:
  print("Creating new Dataset file...")



def write_notes(file_address):
      u = 0
      midi_file = open(file_address, 'rb')
      #print('Processing File:', file_address)
      if encoding_type == 'score-one-byte-encoding':
        score = MIDI.midi2score(midi_file.read())
        midi_file.close()
        # ['note', start_time, duration, channel, note, velocity]

        itrack = 1
        


        notes = []

        tokens = []

        this_channel_has_note = False

        file = open('Dataset.txt', 'a')
        file.write('H d0 tMIDI-TXT-MIDI-Textual-Music-Dataset ')
        while itrack < len(score):
            for event in score[itrack]:

                if event[0] == 'note':
                    this_channel_has_note = True
                    notes.append(event[4])
                    
                    tokens.append([event[5], event[3], event[2], event[1]])
                    file.write('N' + ' d' + str(event[1]) + ' D' + str(event[2]) + ' C' + str(event[3]) + ' n' + str(event[4]) + ' V' + str(event[5]) + ' ')

            itrack += 1
            if not this_channel_has_note:
              u+=1
              if debug: 
                print('Uknown Event: ', event[0])

            if this_channel_has_note and len(notes) > sample_length_in_MIDI_events:
              if enable_sampling:
                break
          

        file.close()
        if debug:
          print('File:', midi_file, 'Number of skipped events: ', u)

      if encoding_type == 'opus-one-byte-encoding':
        score = MIDI.midi2opus(midi_file.read())
        midi_file.close()
        # ['note', start_time, duration, channel, note, velocity]

        itrack = 1


        notes = []

        tokens = []

        this_channel_has_note = False

        file = open('Dataset.txt', 'a')
        file.write('H d0 tMIDI-TXT-MIDI-Textual-Music-Dataset ')
        while itrack < len(score):
            for event in score[itrack]:

                if event[0] == 'note_off':
                    this_channel_has_note = True
                    notes.append(event[3])

                    
                    tokens.append([event[3], event[4], event[1]])
                    
                    file.write('F' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' n' + str(event[3]) + ' v' + str(event[4]) + ' ')


                if event[0] == 'note_on':
                    this_channel_has_note = True
                    notes.append(event[3])
                    
                    tokens.append([event[3], event[4], event[1]])

                    file.write('N' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' n' + str(event[3]) + ' v' + str(event[4]) + ' ')

                if event[0] == 'key_after_touch':
                  if advanced_events:
                    this_channel_has_note = True

                    
                    tokens.append([event[3], event[4], event[1]])
                    file.write('K' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' n' + str(event[3]) + ' v' + str(event[4]) + ' ')

                if event[0] == 'control_change':
                  if advanced_events:
                      if allow_control_change:
                        this_channel_has_note = True

                    
                        tokens.append([event[3], event[4], event[1]])
                    
                        file.write('C' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' r' + str(event[3]) + ' l' + str(event[4]) + ' ')

                if event[0] == 'patch_change':
                  if advanced_events:
                      this_channel_has_note = True
                  
                      tokens.append([event[3], event[2], event[1]])
                    
                      file.write('P' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' h' + str(event[3]) + ' ')

                if event[0] == 'channel_after_touch':
                  if advanced_events:
                      this_channel_has_note = True

                    
                      tokens.append([event[3], event[2], event[1]])
                    
                      file.write('Z' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' v' + str(event[3]) + ' ')

                if event[0] == 'pitch_wheel_change':
                  if advanced_events:
                    this_channel_has_note = True

                    
                    tokens.append([event[3], event[2], event[1]])
                    
                    file.write('W' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' p' + str(event[3]) + ' ')


                if event[0] == 'text_event':
                  if karaoke:
                      this_channel_has_note = True

                      tokens.append([event[2], event[1]])
                      
                      file.write('T' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'copyright_text_event':
                  if karaoke:
                      this_channel_has_note = True
                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('R' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'track_name':
                      this_channel_has_note = True
                    
                      tokens.append([event[2], event[1]])
                      
                      file.write('H' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'instrument_name':
                      this_channel_has_note = True
                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('I' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'lyric':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])
                      file.write('L' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'marker':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('M' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'cue_point':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[3], event[4], event[1]])
                      
                      file.write('U' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_08':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])

                      file.write('+' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_09':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])

                      file.write('&' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0a':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('@' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0b':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('#' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'text_event_0c':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('$' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0d':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('%' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0e':
                  if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('*' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0f':
                  if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('=' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'end_track':
                    this_channel_has_note = True
                    
                    tokens.append([event[1]])                
                    file.write('E' + ' d' + str(event[1]) + ' ')

                if event[0] == 'set_tempo':
                  if advanced_events:
                    if allow_tempo_changes:
                      this_channel_has_note = True
                    
                      tokens.append([ event[2], event[1]])
                      
                      file.write('S' + ' d' + str(event[1]) + ' o' + str(event[2]) + ' ')

                if event[0] == 'smpte_offset':
                  if advanced_events:
                    this_channel_has_note = True
                    
                    tokens.append([event[3], event[4], event[1]])
                    
                    file.write('Y' + ' d' + str(event[1]) + ' g' + str(event[2]) + ' n' + str(event[3]) + ' s' + str(event[4]) + ' f' + str(event[5]) + ' e' + str(event[6]) +' ')

                if event[0] == 'time_signature':
                  if advanced_events:
                    this_channel_has_note = True

                    
                    tokens.append([event[3], event[4], event[1]])
                    
                    file.write('B' + ' d' + str(event[1]) + ' u' + str(event[2]) + ' y' + str(event[3]) + ' i' + str(event[4]) + ' j' + str(event[5]) +' ')


                if event[0] == 'key_signature':
                  if advanced_events:
                    this_channel_has_note = True
                    
                    tokens.append([event[3], event[2], event[1]])
                    
                    file.write('A' + ' d' + str(event[1]) + ' b' + str(event[2]) + ' q' + str(event[3]) + ' ')


                if event[0] == 'sequincer_specific':
                  if advanced_events:
                      this_channel_has_note = True
                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('D' + ' d' + str(event[1]) + ' x' + str(event[2]) + ' ')


                if event[0] == 'raw_meta_event':
                  if advanced_events:
                      this_channel_has_note = True  

                      tokens.append([ event[2], event[1]]) 

                      file.write('E' + ' d' + str(event[1]) + ' z' + str(event[2]) + ' x' + str(event[2]) + ' ')

                if event[0] == 'sysex_f0':
                  if advanced_events:
                      this_channel_has_note = True   

                      tokens.append([ event[2], event[1]])  

                      file.write('G' + ' d' + str(event[1]) + ' x' + str(event[2]) + ' ')

                if event[0] == 'sysex_f7':
                  if advanced_events:
                      this_channel_has_note = True  

                      tokens.append([ event[2], event[1]]) 

                      file.write('!' + ' d' + str(event[1]) + ' x' + str(event[2]) + ' ')
                    
                if event[0] == 'song_position':
                  if advanced_events:
                      this_channel_has_note = True

                      tokens.append([ event[2], event[1]])

                      file.write('J' + ' d' + str(event[1]) + ' a' + str(event[2]) + ' ')

                if event[0] == 'song_select':
                  if advanced_events:
                      this_channel_has_note = True 

                      tokens.append([ event[2], event[1]])

                      file.write('O' + ' d' + str(event[1]) + ' m' + str(event[2]) + ' ')

                if event[0] == 'tune_request':
                  if advanced_events:
                      this_channel_has_note = True

                      tokens.append([ event[2], event[1]])

                      file.write('X' + ' d' + str(event[1]) + ' ')



            itrack += 1
            if not this_channel_has_note:
              print('Uknown Event: ', event[0])

            if this_channel_has_note and len(notes) > sample_length_in_MIDI_events:
              if enable_sampling:
                break         

        file.close()

      if encoding_type == 'opus-complete-words-encoding':

        score = MIDI.midi2opus(midi_file.read())
        midi_file.close()
        # ['note', start_time, duration, channel, note, velocity]

        itrack = 1


        notes = []

        tokens = []

        this_channel_has_note = False

        file = open('Dataset.txt', 'a')
        file.write('H d0 tMIDI-TXT-MIDI-Textual-Music-Dataset ')
        while itrack < len(score):
            for event in score[itrack]:

                if event[0] == 'note_off':
                    this_channel_has_note = True
                    notes.append(event[3])

                    
                    tokens.append([event[3], event[4], event[1]])
                    
                    file.write('NoteOff' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' n' + str(event[3]) + ' v' + str(event[4]) + ' ')


                if event[0] == 'note_on':
                    this_channel_has_note = True
                    notes.append(event[3])
                    
                    tokens.append([event[3], event[4], event[1]])

                    file.write('NoteOn' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' n' + str(event[3]) + ' v' + str(event[4]) + ' ')

                if event[0] == 'key_after_touch':
                  if advanced_events:
                    this_channel_has_note = True

                    
                    tokens.append([event[3], event[4], event[1]])
                    file.write('KeyAfterTouch' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' n' + str(event[3]) + ' v' + str(event[4]) + ' ')

                if event[0] == 'control_change':
                  if advanced_events:
                      if allow_control_change:
                        this_channel_has_note = True

                    
                        tokens.append([event[3], event[4], event[1]])
                    
                        file.write('C' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' r' + str(event[3]) + ' l' + str(event[4]) + ' ')

                if event[0] == 'patch_change':
                  if advanced_events:
                      this_channel_has_note = True
                  
                      tokens.append([event[3], event[2], event[1]])
                    
                      file.write('PatchChange' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' h' + str(event[3]) + ' ')

                if event[0] == 'channel_after_touch':
                  if advanced_events:
                      this_channel_has_note = True

                    
                      tokens.append([event[3], event[2], event[1]])
                    
                      file.write('ChannelAfterTouch' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' v' + str(event[3]) + ' ')

                if event[0] == 'pitch_wheel_change':
                  if advanced_events:
                    this_channel_has_note = True

                    
                    tokens.append([event[3], event[2], event[1]])
                    
                    file.write('PitchWheelChange' + ' d' + str(event[1]) + ' c' + str(event[2]) + ' p' + str(event[3]) + ' ')


                if event[0] == 'text_event':
                  if karaoke:
                      this_channel_has_note = True

                      tokens.append([event[2], event[1]])
                      
                      file.write('TextEvent' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'copyright_text_event':
                  if karaoke:
                      this_channel_has_note = True
                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('CopyrightTextEvent' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'track_name':
                      this_channel_has_note = True
                    
                      tokens.append([event[2], event[1]])
                      
                      file.write('TrackName' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'instrument_name':
                      this_channel_has_note = True
                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('InstrumentName' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'lyric':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])
                      file.write('Lyric' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'marker':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('Marker' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'cue_point':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[3], event[4], event[1]])
                      
                      file.write('CuePoint' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_08':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])

                      file.write('TextEvent08' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_09':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])

                      file.write('TextEvent09' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0a':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('TextEvent0a' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0b':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('TextEvent0b' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')


                if event[0] == 'text_event_0c':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('TextEvent0c' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0d':
                    if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('TextEvent0d' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0e':
                  if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('TextEvent0e' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'text_event_0f':
                  if karaoke:
                      this_channel_has_note = True

                      
                      tokens.append([event[2], event[1]])                  
                      file.write('TextEvent0f' + ' d' + str(event[1]) + ' t' + str(event[2]) + ' ')

                if event[0] == 'end_track':
                    this_channel_has_note = True
                    
                    tokens.append([event[1]])                
                    file.write('EndOfTrack' + ' d' + str(event[1]) + ' ')

                if event[0] == 'set_tempo':
                  if advanced_events:
                    if allow_tempo_changes:
                      this_channel_has_note = True
                    
                      tokens.append([ event[2], event[1]])
                      
                      file.write('SetTempo' + ' d' + str(event[1]) + ' o' + str(event[2]) + ' ')

                if event[0] == 'smpte_offset':
                  if advanced_events:
                    this_channel_has_note = True
                    
                    tokens.append([event[3], event[4], event[1]])
                    
                    file.write('SMPTEOffset' + ' d' + str(event[1]) + ' g' + str(event[2]) + ' n' + str(event[3]) + ' s' + str(event[4]) + ' f' + str(event[5]) + ' e' + str(event[6]) +' ')

                if event[0] == 'time_signature':
                  if advanced_events:
                    this_channel_has_note = True

                    
                    tokens.append([event[3], event[4], event[1]])
                    
                    file.write('TimeSignature' + ' d' + str(event[1]) + ' u' + str(event[2]) + ' y' + str(event[3]) + ' i' + str(event[4]) + ' j' + str(event[5]) +' ')


                if event[0] == 'key_signature':
                  if advanced_events:
                    this_channel_has_note = True
                    
                    tokens.append([event[3], event[2], event[1]])
                    
                    file.write('KeySignature' + ' d' + str(event[1]) + ' b' + str(event[2]) + ' q' + str(event[3]) + ' ')


                if event[0] == 'sequincer_specific':
                  if advanced_events:
                      this_channel_has_note = True
                      
                      tokens.append([event[2], event[1]])
                      
                      file.write('SequencerSpecific' + ' d' + str(event[1]) + ' x' + str(event[2]) + ' ')


                if event[0] == 'raw_meta_event':
                  if advanced_events:
                      this_channel_has_note = True  

                      tokens.append([ event[2], event[1]]) 

                      file.write('RawMetaEvent' + ' d' + str(event[1]) + ' z' + str(event[2]) + ' x' + str(event[2]) + ' ')

                if event[0] == 'sysex_f0':
                  if advanced_events:
                      this_channel_has_note = True   

                      tokens.append([ event[2], event[1]])  

                      file.write('SysExF0' + ' d' + str(event[1]) + ' x' + str(event[2]) + ' ')

                if event[0] == 'sysex_f7':
                  if advanced_events:
                      this_channel_has_note = True  

                      tokens.append([ event[2], event[1]]) 

                      file.write('SysExF7' + ' d' + str(event[1]) + ' x' + str(event[2]) + ' ')
                    
                if event[0] == 'song_position':
                  if advanced_events:
                      this_channel_has_note = True

                      tokens.append([ event[2], event[1]])

                      file.write('SongPosition' + ' d' + str(event[1]) + ' a' + str(event[2]) + ' ')

                if event[0] == 'song_select':
                  if advanced_events:
                      this_channel_has_note = True 

                      tokens.append([ event[2], event[1]])

                      file.write('SongSelect' + ' d' + str(event[1]) + ' m' + str(event[2]) + ' ')

                if event[0] == 'tune_request':
                  if advanced_events:
                      this_channel_has_note = True

                      tokens.append([ event[2], event[1]])

                      file.write('TuneRequest' + ' d' + str(event[1]) + ' ')



            itrack += 1
            if not this_channel_has_note:
              print('Uknown Event: ', event[0])

            if this_channel_has_note and len(notes) > sample_length_in_MIDI_events:
              if enable_sampling:
                break
          

        file.close()      
       

dataset_addr = "Dataset"
files = os.listdir(dataset_addr)
for file in tqdm.auto.tqdm(files):
    path = os.path.join(dataset_addr, file)
    write_notes(path)
#print('Done!')
#print('Number of skipped events: ', u)
#@title Define Constants and Functions { run: "auto" }
number_of_training_batches = 128 #@param {type:"slider", min:0, max:128, step:4}
attention_sequence_length = 256 #@param {type:"slider", min:0, max:512, step:16}
embedding_size = 256 #@param {type:"slider", min:0, max:1024, step:16}
LSTM_layers_size = 256 #@param {type:"slider", min:0, max:1024, step:16}
full_path_to_txt_dataset = "/content/Dataset.txt" #@param {type:"string"}

import numpy as np
import tensorflow as tf
import os

import distutils
if distutils.version.LooseVersion(tf.__version__) < '2.0':
    raise Exception('This notebook is compatible with TensorFlow 2.0 or higher.')

INPUT_TXT = full_path_to_txt_dataset

def transform(txt):
  return np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)

def input_fn(seq_len=attention_sequence_length, batch_size=number_of_training_batches):
  """Return a dataset of source and target sequences for training."""
  with tf.io.gfile.GFile(INPUT_TXT, 'r') as f:
    txt = f.read()

  source = tf.constant(transform(txt), dtype=tf.int32)

  ds = tf.data.Dataset.from_tensor_slices(source).batch(seq_len+1, drop_remainder=True)

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

  BUFFER_SIZE = 10000
  ds = ds.map(split_input_target).shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

  return ds.repeat()


EMBEDDING_DIM = LSTM_layers_size

def lstm_model(seq_len=attention_sequence_length, batch_size=None, stateful=True):
  """Language model: predict the next word given the current word."""
  source = tf.keras.Input(
      name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

  embedding = tf.keras.layers.Embedding(input_dim=embedding_size, output_dim=EMBEDDING_DIM)(source)
  lstm_1 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(embedding)
  lstm_2 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(lstm_1)
  lstm_3 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(lstm_2)
  predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(LSTM_layers_size, activation='softmax'))(lstm_3)
  return tf.keras.Model(inputs=[source], outputs=[predicted_char])
#@title Train the Model
number_of_training_epochs = 50 #@param {type:"slider", min:1, max:50, step:1}
num_steps_per_epoch = 100 #@param {type:"slider", min:0, max:1000, step:10}
model_learning_rate = 0.01 #@param {type:"slider", min:0, max:0.01, step:0.0001}
save_every_number_of_steps = 500 #@param {type:"slider", min:0, max:1000, step:10}
save_only_best_checkpoints = True #@param {type:"boolean"}


from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
checkpoint = ModelCheckpoint(
        'MIDI-TXT-MIDI-TPU-Model.h5',
        save_freq=save_every_number_of_steps, #Every # epochs
        monitor='loss',
        verbose=1,
        save_best_only=save_only_best_checkpoints,
        mode='min'
    )


tf.keras.backend.clear_session()

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
  training_model = lstm_model(seq_len=attention_sequence_length, stateful=False)
  training_model.compile(
      optimizer=tf.keras.optimizers.RMSprop(learning_rate=model_learning_rate),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])

history = training_model.fit(
    input_fn(),
    steps_per_epoch=num_steps_per_epoch,
    epochs=number_of_training_epochs,
    callbacks = [checkpoint]
)
training_model.save_weights('/content/MIDI-TXT-MIDI-TPU-Model.h5', overwrite=True)

#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.title('MIDI-TXT-MIDI Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('MIDI-TXT-MIDI Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
#@title Save/Re-Save the Model from memory if needed
training_model.save_weights('/content/MIDI-TXT-MIDI-TPU-Model.h5')
#@title Load/Reload the Model from saved checkpoint if needed
training_model.load_weights('/content/MIDI-TXT-MIDI-TPU-Model.h5')
#@title Generate Output
number_of_tokens_to_generate = 2048 #@param {type:"slider", min:0, max:16384, step:128}
input_model_priming_sequence = "N" #@param {type:"string"}
number_of_output_batches = 1 #@param {type:"slider", min:1, max:16, step:1}
debug = False #@param {type:"boolean"}

import tqdm.auto

BATCH_SIZE = number_of_output_batches
PREDICT_LEN = number_of_tokens_to_generate

# Keras requires the batch size be specified ahead of time for stateful models.
# We use a sequence length of 1, as we will be feeding in one character at a 
# time and predicting the next character.
prediction_model = lstm_model(seq_len=1, batch_size=BATCH_SIZE, stateful=True)
prediction_model.load_weights('/content/MIDI-TXT-MIDI-TPU-Model.h5')

# We seed the model with our initial string, copied BATCH_SIZE times

seed_txt = input_model_priming_sequence
seed = transform(seed_txt)
seed = np.repeat(np.expand_dims(seed, 0), BATCH_SIZE, axis=0)

if debug: print('Generating Batches...')
# First, run the seed forward to prime the state of the model.
prediction_model.reset_states()
for i in range(len(seed_txt) - 1):
  prediction_model.predict(seed[:, i:i + 1])

if debug: print('Accumulating predictions...')
# Now we can accumulate predictions!
predictions = [seed[:, -1:]]
for i in tqdm.auto.tqdm(range(PREDICT_LEN)):
  last_word = predictions[-1]
  next_probits = prediction_model.predict(last_word)[:, 0, :]
  
  #if debug: print('Sampling from output distribution...') 
  # sample from our output distribution
  next_idx = [
      np.random.choice(256, p=next_probits[i])
      for i in range(BATCH_SIZE)
  ]
  predictions.append(np.asarray(next_idx, dtype=np.int32))
  
if debug: print('Generating Batches...')
for i in tqdm.auto.tqdm(range(BATCH_SIZE)):
  if debug:print('PREDICTION %d\n\n' % i)
  p = [predictions[j][i] for j in range(PREDICT_LEN)]
  generated = ''.join([chr(c) for c in p])  # Convert back to text
  if debug: print(generated)
  if debug: print()
  assert len(generated) == PREDICT_LEN, 'Generated text too short'
  file_nm = '/content/output-' + str(i) + '.txt'
  with open(file_nm, 'w') as gen_song_file:
    gen_song_file.write(generated)

from google.colab import files
files.download('/content/output-0.txt')
#@title Convert to MIDI from TXT
number_of_ticks_per_quarter = 424 #@param {type:"slider", min:0, max:1280, step:8}
output_batch_number_to_convert_to_MIDI = 0 #@param {type:"slider", min:0, max:15, step:1}

import MIDI
import tqdm.auto
notes = []
velocities = []
timings = []
durations = []

batch_nm = '/content/output-' + str(output_batch_number_to_convert_to_MIDI) + '.txt'

with open(batch_nm, 'r') as file:
    notestring=file.read()

score_note = notestring.split(" ")

score = score_note

i=0

z=0

zero_marker = True

song_score = [number_of_ticks_per_quarter, 
              [['track_name', 0, b'Composed by Artificial Intelligence Model']],              
              ]
if karaoke:
  song_score.append([['track_name', 0, b'M-T-M 3.x Karaoke Encoding']])
else:
  song_score.append([['track_name', 0, b'M-T-M 3.x Music Encoding']])

for i in tqdm.auto.tqdm(range(len(score))):

        # if the event is a blank, space, "eos" or unknown, skip and go to next event
        if score[i] in ["", " ", "<eos>", "<unk>"]:
            continue

        # if the event starts with 'end' indicating an end of note
        elif score[i][:2]=="@@":

            continue

        # in this block, we are looking for notes   
        else:
            # Look ahead to see if an end<noteid> was generated
            # soon after.  


            note_string_len = len(score[i])
            for j in range(1,200):
                if i+j==len(score):
                    break
            if encoding_type == 'score-one-byte-encoding':
              if score[i] == 'N':
                try:
                  if zero_marker == True:
                    trk_nm = 'Track #' + str(z++1)
                    song_score.append([['track_name', 0, trk_nm]])
                    zero_marker = False
                  song_score[-1].append(['note', 
                                        int(score[i+1][1:]), #Start Time
                                        int(score[i+2][1:]), #Duration
                                        int(score[i+3][1:]), #Channel
                                        int(score[i+4][1:]), #Note
                                        int(score[i+5][1:])]) #Velocity
                            
                except:
                  print("Unknown event: " + score[i] + ' ' + score[i+1])

            if encoding_type == 'opus-one-byte-encoding':
              if score[i] == 'F':
                try:
                  song_score[-1].append(['note_off', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:])])
                except:
                  print("Unknown event: " + score[i] + ' ' + score[i+1])

              if score[i] == 'N':
                try:
                  if zero_marker == True:
                    trk_nm = 'Track #' + str(z++1)
                    song_score.append([['track_name', 0, trk_nm]])
                    zero_marker = False
                  song_score[-1].append(['note_on', 
                                        int(score[i+1][1:]), #Duration
                                        int(score[i+2][1:]), #Channel
                                        int(score[i+3][1:]), #Note
                                        int(score[i+4][1:])]) #Velocity
                            
                except:
                  print("Unknown event: " + score[i] + ' ' + score[i+1])

              if score[i] == 'K':
                try:
                  song_score[-1].append(['key_after_touch', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:])])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'C':
                try:
                  song_score[-1].append(['control_change',
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]), #Controller
                                        int(score[i+4][1:])]) #ControlValue
                except:
                  print("Unknown event: " + score[i])




              if score[i] == 'P':
                try:
                  song_score[-1].append(['patch_change', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])
                                        ])
                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'Z':
                try:
                  song_score[-1].append(['channel_after_touch', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])
                                        ])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'W':
                try:
                  song_score[-1].append(['pitch_wheel_change', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])
                                        ])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'T':
                try:
                  song_score[-1].append(['text_event', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                  zero_marker = True
                except:
                  print("Unknown event: " + score[i] + ' ' + score[i+1] + ' ' + score[i+2])

              if score[i] == 'R':
                try:
                  song_score[-1].append(['copyright_text_event', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'H':
                try:
                  song_score[-1].append(['track_name', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                  zero_marker = True
                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'I':
                try:
                  song_score[-1].append(['instrument_name', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])              
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'L':
                try:
                  song_score[-1].append(['lyric', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                  zero_marker = True
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'M':
                try:
                  song_score[-1].append(['marker', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'U':
                try:
                  song_score[-1].append(['cue_point', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '+':
                try:
                  song_score[-1].append(['text_event_08', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '&':
                try:
                  song_score[-1].append(['text_event_09', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '@':
                try:
                  song_score[-1].append(['text_event_0a', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '#':
                try:
                  song_score[-1].append(['text_event_0b', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '$':
                try:
                  song_score[-1].append(['text_event_0c', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '%':
                try:
                  song_score[-1].append(['text_event_0d', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '*':
                try:
                  song_score[-1].append(['text_event_0e', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == '=':
                try:
                  song_score[-1].append(['text_event_0f', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'E':
                try:
                  song_score[-1].append(['end_track', 
                                        int(score[i+1][1:])])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'S':
                try:
                  song_score[-1].append(['set_tempo', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'Y':
                try:
                  song_score[-1].append(['smpte_offset',
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:]),
                                        int(score[i+5][1:]),
                                        int(score[i+6][1:])])
                except:
                  print("Unknown event: " + score[i])                

              if score[i] == 'B':
                try:
                  song_score[-1].append(['time_signature', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:]),
                                        int(score[i+5][1:])])

                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'A':
                try:
                  song_score[-1].append(['key_signature', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])])

                except:
                  print("Unknown event: " + score[i])



              if score[i] == 'D':
                try:
                  song_score[-1].append(['sequencer_specific', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'E':
                try:
                  song_score[-1].append(['raw_meta_event', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'G':
                try:
                  song_score[-1].append(['sysex_f0', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == '!':
                try:
                  song_score[-1].append(['sysex_f7', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'J':
                try:
                  song_score[-1].append(['song_position', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'O':
                try:
                  song_score[-1].append(['song_select', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'X':
                try:
                  song_score[-1].append(['tune_request', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])


            if encoding_type == 'opus-complete-words-encoding':
              if score[i] == 'NoteOff':
                try:
                  song_score[-1].append(['note_off', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:])])
                except:
                  print("Unknown event: " + score[i] + ' ' + score[i+1])

              if score[i] == 'NoteOn':
                try:
                  if zero_marker == True:
                    trk_nm = 'Track #' + str(z++1)
                    song_score.append([['track_name', 0, trk_nm]])
                    zero_marker = False
                  song_score[-1].append(['note_on', 
                                        int(score[i+1][1:]), #Duration
                                        int(score[i+2][1:]), #Channel
                                        int(score[i+3][1:]), #Note
                                        int(score[i+4][1:])]) #Velocity
                            
                except:
                  print("Unknown event: " + score[i] + ' ' + score[i+1])

              if score[i] == 'KeyAfterTouch':
                try:
                  song_score[-1].append(['key_after_touch', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:])])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'ControlChange':
                try:
                  song_score[-1].append(['control_change',
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]), #Controller
                                        int(score[i+4][1:])]) #ControlValue
                except:
                  print("Unknown event: " + score[i])




              if score[i] == 'PatchChange':
                try:
                  song_score[-1].append(['patch_change', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])
                                        ])
                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'ChannelAfterTouch':
                try:
                  song_score[-1].append(['channel_after_touch', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])
                                        ])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'PitchWheelChange':
                try:
                  song_score[-1].append(['pitch_wheel_change', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])
                                        ])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent':
                try:
                  song_score[-1].append(['text_event', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                  zero_marker = True
                except:
                  print("Unknown event: " + score[i] + ' ' + score[i+1] + ' ' + score[i+2])

              if score[i] == 'CopyrightTextEvent':
                try:
                  song_score[-1].append(['copyright_text_event', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TrackName':
                try:
                  song_score[-1].append(['track_name', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                  zero_marker = True
                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'InstrumentName':
                try:
                  song_score[-1].append(['instrument_name', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])              
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'Lyric':
                try:
                  song_score[-1].append(['lyric', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                  zero_marker = True
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'Marker':
                try:
                  song_score[-1].append(['marker', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'CuePoint':
                try:
                  song_score[-1].append(['cue_point', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent08':
                try:
                  song_score[-1].append(['text_event_08', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent09':
                try:
                  song_score[-1].append(['text_event_09', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent0a':
                try:
                  song_score[-1].append(['text_event_0a', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent0b':
                try:
                  song_score[-1].append(['text_event_0b', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent0c':
                try:
                  song_score[-1].append(['text_event_0c', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent0d':
                try:
                  song_score[-1].append(['text_event_0d', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent0e':
                try:
                  song_score[-1].append(['text_event_0e', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TextEvent0f':
                try:
                  song_score[-1].append(['text_event_0f', 
                                        int(score[i+1][1:]), 
                                        score[i+2][1:]])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'EndOfTrack':
                try:
                  song_score[-1].append(['end_track', 
                                        int(score[i+1][1:])])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'SetTempo':
                try:
                  song_score[-1].append(['set_tempo', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])
                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'SMPTEOffset':
                try:
                  song_score[-1].append(['smpte_offset',
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:]),
                                        int(score[i+5][1:]),
                                        int(score[i+6][1:])])
                except:
                  print("Unknown event: " + score[i])                

              if score[i] == 'TimeSignature':
                try:
                  song_score[-1].append(['time_signature', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:]),
                                        int(score[i+4][1:]),
                                        int(score[i+5][1:])])

                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'KeySignature':
                try:
                  song_score[-1].append(['key_signature', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:]),
                                        int(score[i+3][1:])])

                except:
                  print("Unknown event: " + score[i])



              if score[i] == 'SequencerSpecific':
                try:
                  song_score[-1].append(['sequencer_specific', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'RawMetaEvent':
                try:
                  song_score[-1].append(['raw_meta_event', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'SysEx_F0':
                try:
                  song_score[-1].append(['sysex_f0', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'SysEx_F7':
                try:
                  song_score[-1].append(['sysex_f7', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])


              if score[i] == 'SongPosition':
                try:
                  song_score[-1].append(['song_position', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'SongSelect':
                try:
                  song_score[-1].append(['song_select', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

              if score[i] == 'TuneRequest':
                try:
                  song_score[-1].append(['tune_request', 
                                        int(score[i+1][1:]), 
                                        int(score[i+2][1:])])

                except:
                  print("Unknown event: " + score[i])

if encoding_type == 'score-one-byte-encoding':
  midi_data = MIDI.score2midi(song_score)
  if debug:
    print('Encoding Type: ', encoding_type)
else:
  midi_data = MIDI.opus2midi(song_score)
  if debug:
    print('Encoding Type: ', encoding_type)

with open('output.mid', 'wb') as midi_file:
    midi_file.write(midi_data)
    midi_file.close()
print('Done!')

from google.colab import files
files.download('/content/output.mid')

MIDI.score2stats(song_score)
#@title Download all TXT output files/batches as one nice Zip archive
!zip 'MIDI-TXT-MIDI-All-Output-Batches.zip' out*.txt
from google.colab import files
files.download('MIDI-TXT-MIDI-All-Output-Batches.zip')