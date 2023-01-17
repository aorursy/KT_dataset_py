import re

def pp(filtered_output, real_tweet):

    filtered_output = ' '.join(filtered_output.split())

    if len(real_tweet.split()) < 2:

        filtered_output = real_tweet

    else:

        if len(filtered_output.split()) == 1:

            if filtered_output.endswith(".."):

                if real_tweet.startswith(" "):

                    st = real_tweet.find(filtered_output)

                    fl = real_tweet.find("  ")

                    if fl != -1 and fl < st:

                        filtered_output = re.sub(r'(\.)\1{2,}', '', filtered_output)

                    else:

                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)

                else:

                    st = real_tweet.find(filtered_output)

                    fl = real_tweet.find("  ")

                    if fl != -1 and fl < st:

                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)

                    else:

                        filtered_output = re.sub(r'(\.)\1{2,}', '..', filtered_output)

                return filtered_output

            if filtered_output.endswith('!!'):

                if real_tweet.startswith(" "):

                    st = real_tweet.find(filtered_output)

                    fl = real_tweet.find("  ")

                    if fl != -1 and fl < st:

                        filtered_output = re.sub(r'(\!)\1{2,}', '', filtered_output)

                    else:

                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)

                else:

                    st = real_tweet.find(filtered_output)

                    fl = real_tweet.find("  ")

                    if fl != -1 and fl < st:

                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)

                    else:

                        filtered_output = re.sub(r'(\!)\1{2,}', '!!', filtered_output)

                return filtered_output



        if real_tweet.startswith(" "):

            filtered_output = filtered_output.strip()

            text_annotetor = ' '.join(real_tweet.split())

            start = text_annotetor.find(filtered_output)

            end = start + len(filtered_output)

            start -= 0

            end += 2

            flag = real_tweet.find("  ")

            if flag < start:

                filtered_output = real_tweet[start:end]



        if "  " in real_tweet and not real_tweet.startswith(" "):

            filtered_output = filtered_output.strip()

            text_annotetor = re.sub(" {2,}", " ", real_tweet)

            start = text_annotetor.find(filtered_output)

            end = start + len(filtered_output)

            start -= 0

            end += 2

            flag = real_tweet.find("  ")

            if flag < start:

                filtered_output = real_tweet[start:end]

    return filtered_output
tweet = "  ROFLMAO for the funny web portal  =D"

pred = "funny"

answer = "e funny"

pp(pred, tweet)
tweet = " yea i just got outta one too....i want him back tho  but i feel the same way...i`m cool on dudes for a lil while"

pred = "cool"

answer = "m cool"

pp(pred, tweet)
tweet = "Ow... My shoulder muscle (I can`t remember the name :p) hurts... What did I do?  I don`t even know"

pred = "hurts..."

answer = "hurts.."

pp(pred, tweet)