def number_of_lines():

    file = open('../input/churn-emails-mbox/mbox.txt')

    data = file.read()

    count=0

    for i in data:

        if i == '\n':

            count += 1

    file.close()

    return count
print(number_of_lines())
def count_number_of_lines():

    count=0

    with open('../input/churn-emails-mbox/mbox.txt') as file:

        for line in file:

            line = line.rstrip()

            if line.startswith('Subject:'):

                count+=1

    file.close()

    return count
print(count_number_of_lines())
def average_spam_confidence():

    with open('../input/churn-emails-mbox/mbox.txt') as file:

        l = []

        count = 0

        for line in file:

            line = line.strip()

            if line.startswith('X-DSPAM-Confidence'):

                var, num = line.split(':')

                l.append(float(num))

                count += 1

        file.close()

        avg = sum(l) / count

        return avg

print(average_spam_confidence())
def find_email_sent_days():

    with open('../input/churn-emails-mbox/mbox.txt') as file:

        day = dict()

        for line in file:

            line = line.strip()

            if line.startswith('From') and not line.startswith('From:'):

                List = line.split()

                if List[2] in day:

                    day[str(List[2])] += 1

                else:

                    day[str(List[2])] = 1

        return day
print(find_email_sent_days())
def count_message_from_email():

    with open('../input/churn-emails-mbox/mbox.txt') as file:

        email_addresses = dict()

        for line in file:

            line = line.strip()

            if line.startswith('From') and not line.startswith('From:'):

                List = line.split()

                if List[1] in email_addresses:

                    email_addresses[str(List[1])] += 1

                else:

                    email_addresses[str(List[1])] = 1

        return email_addresses
print(count_message_from_email())
def count_message_from_domain():

    with open('../input/churn-emails-mbox/mbox.txt') as file:

        domain = dict()

        for line in file:

            line = line.strip()

            if line.startswith('From') and not line.startswith('From:'):

                List = line.split()

                dom = List[1].split('@')[1]

                if dom in domain:

                    domain[dom] += 1

                else:

                    domain[dom] = 1

        return domain
print(count_message_from_domain())
def find_most_used_domain():

    domain = count_message_from_domain()

    max_used =  max(domain, key=domain.get) 

    return max_used, domain[max_used]
print(find_most_used_domain())