#!pip install googletrans

#!pip install google-cloud-speech

#!pip install --upgrade python-speech

#!pip install pyaudio

#!pip install google-cloud-translate==2.0.0
from __future__ import division

import googletrans 

from googletrans import Translator

import re

import sys

from google.cloud import speech

from google.cloud.speech import enums

from google.cloud.speech import types

import pyaudio

from six.moves import queue

RATE = 16000

CHUNK = int(RATE / 10)  # 100ms



class MicrophoneStream(object):

    def __init__(self, rate, chunk):

        self._rate = rate

        self._chunk = chunk

        self._buff = queue.Queue()

        self.closed = True



    def __enter__(self):

        self._audio_interface = pyaudio.PyAudio()

        self._audio_stream = self._audio_interface.open(

            format=pyaudio.paInt16,

            channels=1, rate=self._rate,

            input=True, frames_per_buffer=self._chunk,

            stream_callback=self._fill_buffer,

        )

        self.closed = False

        return self



    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()

        self._audio_stream.close()

        self.closed = True

        self._buff.put(None)

        self._audio_interface.terminate()



    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):

        self._buff.put(in_data)

        return None, pyaudio.paContinue



    def generator(self):

        while not self.closed:

            chunk = self._buff.get()

            if chunk is None:

                return

            data = [chunk]

            while True:

                try:

                    chunk = self._buff.get(block=False)

                    if chunk is None:

                        return

                    data.append(chunk)

                except queue.Empty:

                    break

            yield b''.join(data)



def listen_print_loop(responses):

    num_chars_printed = 0

    #num_translated_chars_printed = 0

    for response in responses:

        if not response.results:

            continue

        result = response.results[0]

        if not result.alternatives:

            continue

        transcript = result.alternatives[0].transcript

        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            

        translator = Translator()

        translator_overwrite_chars = Translator()



        if not result.is_final:

            sys.stdout.write(transcript + overwrite_chars + '\r')

            sys.stdout.flush()

            num_chars_printed = len(transcript)



        else:

            translated_transcript=translator.translate(transcript + overwrite_chars, src='ar', dest='uz')

            print(transcript + overwrite_chars)

            print(translated_transcript.text)

            if re.search(r'\b(exit|quit|كويت|chiq|yopil|stop|خروج|أغلق)\b', transcript, re.I):

                print('Exiting...')

                break

            num_chars_printed = 0



def main():

    language_code = 'ar-AE'  # a BCP-47 language tag en-US,  ar-AE Arabic (United Arab Emirates),  ar-SA Arabic (Saudi Arabia)

    client = speech.SpeechClient()

    config = types.RecognitionConfig(encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=RATE, language_code=language_code)

    streaming_config = types.StreamingRecognitionConfig(config=config, interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:

        audio_generator = stream.generator()

        requests = (types.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        listen_print_loop(responses)



if __name__ == '__main__':

    main()